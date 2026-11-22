//! Record the inputs the existing synchronous parser actually consults,
//! including failed lookups. Recheck them after parsing and before every hit.
//! A scoped recorder lets shared helpers (including patch-line recovery) report
//! their I/O without giving the other parser and watcher callers a cache API.

use super::{claude, codex, AgentType, Key};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::mem::{size_of, size_of_val};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, PartialEq, Eq)]
enum Stamp {
    Missing,
    Present {
        is_dir: bool,
        len: u64,
        modified: SystemTime,
        // Unix device/inode + ctime, or Windows volume/file index + ChangeTime.
        identity: (u64, u128),
        changed: (i64, i64),
    },
}

#[cfg(unix)]
fn from_metadata(meta: fs::Metadata) -> Option<Stamp> {
    use std::os::unix::fs::MetadataExt;
    // Some virtual/network filesystems supply no useful inode or only
    // whole-second timestamps. A same-size rewrite there is unversionable.
    if meta.ino() == 0 || (meta.ctime_nsec() == 0 && meta.mtime_nsec() == 0) {
        return None;
    }
    (meta.is_file() || meta.is_dir()).then_some(Stamp::Present {
        is_dir: meta.is_dir(),
        len: meta.len(),
        modified: meta.modified().ok()?,
        identity: (meta.dev(), u128::from(meta.ino())),
        changed: (meta.ctime(), meta.ctime_nsec()),
    })
}

#[cfg(unix)]
fn stamp_path(path: &Path) -> io::Result<Option<Stamp>> {
    fs::metadata(path).map(from_metadata)
}

#[cfg(unix)]
fn stamp_file(file: &File) -> Option<Stamp> {
    from_metadata(file.metadata().ok()?)
}

#[cfg(windows)]
fn stamp_path(path: &Path) -> io::Result<Option<Stamp>> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_BACKUP_SEMANTICS, FILE_READ_ATTRIBUTES,
    };
    fs::OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
        .map(|f| stamp_file(&f))
}

#[cfg(windows)]
fn stamp_file(file: &File) -> Option<Stamp> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FileBasicInfo, FileIdInfo, GetFileInformationByHandleEx, FILE_BASIC_INFO, FILE_ID_INFO,
    };
    let meta = file.metadata().ok()?;
    if !meta.is_dir() && !meta.is_file() {
        return None;
    }
    let handle = file.as_raw_handle();
    // SAFETY: both queries borrow the live file handle and write exactly the
    // requested POD structure before assume_init.
    let (info, basic) = unsafe {
        let mut info = std::mem::MaybeUninit::<FILE_ID_INFO>::uninit();
        let mut basic = std::mem::MaybeUninit::<FILE_BASIC_INFO>::uninit();
        if GetFileInformationByHandleEx(
            handle,
            FileIdInfo,
            info.as_mut_ptr().cast(),
            size_of::<FILE_ID_INFO>() as u32,
        ) == 0
            || GetFileInformationByHandleEx(
                handle,
                FileBasicInfo,
                basic.as_mut_ptr().cast(),
                size_of::<FILE_BASIC_INFO>() as u32,
            ) == 0
        {
            return None;
        }
        (info.assume_init(), basic.assume_init())
    };
    let index = u128::from_le_bytes(info.FileId.Identifier);
    (index != 0 && basic.ChangeTime % 10_000_000 != 0).then_some(Stamp::Present {
        is_dir: meta.is_dir(),
        len: meta.len(),
        modified: meta.modified().ok()?,
        identity: (info.VolumeSerialNumber, index),
        changed: (basic.ChangeTime, 0),
    })
}

#[cfg(not(any(unix, windows)))]
fn stamp_path(_: &Path) -> io::Result<Option<Stamp>> {
    Ok(None)
}

#[cfg(not(any(unix, windows)))]
fn stamp_file(_: &File) -> Option<Stamp> {
    None
}

fn stamp(path: &Path) -> Option<Stamp> {
    match stamp_path(path) {
        Ok(value) => value,
        Err(err) if err.kind() == io::ErrorKind::NotFound => Some(Stamp::Missing),
        _ => None,
    }
}

impl Stamp {
    fn recently_changed(&self) -> bool {
        let Self::Present { changed, .. } = self else {
            return false;
        };
        #[cfg(windows)]
        let seconds = changed.0.div_euclid(10_000_000) - 11_644_473_600;
        #[cfg(not(windows))]
        let seconds = changed.0;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        i128::from(now) - i128::from(seconds) <= 2
    }

    fn digest(&self, path: &Path) -> Option<[u8; 32]> {
        let Self::Present { is_dir, .. } = self else {
            return None;
        };
        let mut hash = Sha256::new();
        if *is_dir {
            // Preserve enumeration order: equal-timestamp rollover candidates
            // are resolved in that order.
            for entry in fs::read_dir(path).ok()? {
                let name = entry.ok()?.file_name();
                let bytes = name.as_encoded_bytes();
                hash.update(bytes.len().to_le_bytes());
                hash.update(bytes);
            }
        } else {
            let mut file = File::open(path).ok()?;
            if stamp_file(&file).as_ref() != Some(self) {
                return None;
            }
            let mut buffer = [0; 32 * 1024];
            loop {
                let n = file.read(&mut buffer).ok()?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
        }
        Some(hash.finalize().into())
    }
}

struct Fingerprint {
    stamp: Stamp,
    // The kernel can give two writes in one clock tick identical nanosecond
    // mtimes AND ctimes. Keep a content witness for a recently changed input
    // until a matching read after that tick is safely past (Git's racy-stat
    // problem). Settled long transcripts then validate with stat alone.
    digest: Option<[u8; 32]>,
    settled: AtomicBool,
}

impl Fingerprint {
    fn matches(&self, path: &Path) -> bool {
        if stamp(path).as_ref() != Some(&self.stamp) {
            return false;
        }
        let Some(expected) = self.digest else {
            return true;
        };
        if self.settled.load(Ordering::Relaxed) {
            return true;
        }
        let mature = !self.stamp.recently_changed();
        if self.stamp.digest(path) != Some(expected) {
            return false;
        }
        if mature {
            self.settled.store(true, Ordering::Relaxed);
        }
        true
    }
}

#[derive(Default)]
struct Recorder {
    main: Option<PathBuf>,
    inherited: BTreeMap<String, Option<PathBuf>>,
    paths: BTreeMap<PathBuf, Fingerprint>,
    invalid: bool,
}

thread_local! {
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

fn recording(f: impl FnOnce(&mut Recorder)) {
    RECORDER.with_borrow_mut(|slot| {
        if let Some(recorder) = slot.as_mut().filter(|r| !r.invalid) {
            f(recorder);
        }
    });
}

struct Restore(Option<Recorder>);
impl Drop for Restore {
    fn drop(&mut self) {
        RECORDER.set(self.0.take());
    }
}

pub(super) fn record<T>(parse: impl FnOnce() -> T) -> (T, Option<Version>) {
    let restore = Restore(RECORDER.replace(Some(Recorder::default())));
    let result = parse();
    let recorder = RECORDER.take().unwrap();
    drop(restore);
    let version = recorder
        .main
        .filter(|_| !recorder.invalid)
        .map(|main| Version {
            main,
            inherited: recorder.inherited.into_iter().collect(),
            paths: recorder.paths.into_iter().collect(),
        });
    (result, version)
}

impl Recorder {
    fn remember(&mut self, path: &Path, token: Option<Stamp>) {
        let Some(token) = token.filter(|_| path.is_absolute()) else {
            self.invalid = true;
            return;
        };
        if let Some(before) = self.paths.get(path) {
            self.invalid = before.stamp != token
                || before
                    .digest
                    .is_some_and(|expected| token.digest(path) != Some(expected));
            return;
        }
        let digest = if token.recently_changed() {
            let Some(value) = token.digest(path) else {
                self.invalid = true;
                return;
            };
            Some(value)
        } else {
            None
        };
        self.paths.insert(
            path.to_path_buf(),
            Fingerprint {
                stamp: token,
                digest,
                settled: AtomicBool::new(false),
            },
        );
    }
}

pub(in crate::parsers) fn selected_main(path: &Path) {
    recording(|r| r.main = Some(path.to_path_buf()));
}

pub(in crate::parsers) fn selected_inherited(id: &str, path: Option<&Path>) {
    recording(|r| {
        r.inherited
            .insert(id.to_owned(), path.map(Path::to_path_buf));
    });
}

pub(in crate::parsers) fn observe(path: &Path) {
    recording(|r| r.remember(path, stamp(path)));
}

pub(in crate::parsers) fn io_error(_: &io::Error) {
    unversioned();
}

pub(in crate::parsers) fn unversioned() {
    recording(|r| r.invalid = true);
}

pub(in crate::parsers) fn unversioned_now() -> chrono::DateTime<chrono::Utc> {
    unversioned();
    chrono::Utc::now()
}

pub(in crate::parsers) fn open(path: &Path) -> io::Result<File> {
    let file = File::open(path);
    recording(|r| {
        // Use the handle's identity. A rename can replace the path between
        // choosing the file and opening it.
        let token = match &file {
            Ok(f) => stamp_file(f),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Some(Stamp::Missing),
            _ => None,
        };
        r.remember(path, token);
    });
    file
}

pub(in crate::parsers) fn read_dir(
    path: &Path,
) -> io::Result<impl Iterator<Item = io::Result<fs::DirEntry>>> {
    observe(path);
    fs::read_dir(path)
        .inspect_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                recording(|r| r.remember(path, Some(Stamp::Missing)));
            } else {
                io_error(e);
            }
        })
        .map(|entries| entries.map(|e| e.inspect_err(io_error)))
}

pub(super) struct Version {
    main: PathBuf,
    inherited: Box<[(String, Option<PathBuf>)]>,
    paths: Box<[(PathBuf, Fingerprint)]>,
}

impl Version {
    pub(super) fn current(&self, key: &Key) -> bool {
        // Repeat the parser's filename selection: a new rollout can supersede
        // the main file or satisfy a history pointer anywhere in sessions/.
        let main = match key.agent {
            AgentType::Codex => {
                let parser = codex::CodexParser::at_sessions_dir(key.root.clone());
                if !self
                    .inherited
                    .iter()
                    .all(|(id, path)| parser.find_rollout_by_rollout_id(id) == *path)
                {
                    return false;
                }
                parser.conversation_path(&key.id)
            }
            AgentType::ClaudeCode => claude::find_session_file_in(&key.root, &key.id),
            _ => return false,
        };
        main.as_ref() == Some(&self.main)
            && self.paths.iter().all(|(path, token)| token.matches(path))
    }

    pub(super) fn bytes(&self) -> u64 {
        size_of::<Self>() as u64
            + self.main.capacity() as u64
            + size_of_val(&*self.paths) as u64
            + size_of_val(&*self.inherited) as u64
            + self
                .paths
                .iter()
                .map(|(p, _)| p.capacity() as u64 + 32)
                .sum::<u64>()
            + self
                .inherited
                .iter()
                .map(|(id, path)| {
                    id.capacity() as u64 + path.as_ref().map_or(0, PathBuf::capacity) as u64 + 64
                })
                .sum::<u64>()
    }
}
