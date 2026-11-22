//! Shared, route-sanitized parser output. A key owns a parse slot only while
//! validating or loading. Disk I/O and the full parse run outside the store lock.

mod sources;
mod weight;

use std::collections::BTreeMap;
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use super::{build_agent_parser, claude, codex, AgentEnv, ParseError};
use crate::models::{AgentType, ConversationDetail};

pub(super) use sources::{
    io_error, observe, open, read_dir, selected_inherited, selected_main, unversioned,
    unversioned_now,
};

const CACHE_BYTES: u64 = 128 * 1024 * 1024;

pub(crate) fn shared_conversation_detail(
    agent_type: AgentType,
    env: &AgentEnv,
    conversation_id: &str,
) -> Result<Arc<ConversationDetail>, ParseError> {
    let parse = || build_agent_parser(agent_type, env).get_conversation(conversation_id);
    let Some(key) = Key::new(agent_type, env, conversation_id) else {
        return parse().map(Arc::new);
    };
    static CACHE: OnceLock<DetailCache> = OnceLock::new();
    CACHE
        .get_or_init(|| DetailCache::new(CACHE_BYTES))
        .get(key, parse)
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    agent: AgentType,
    root: PathBuf,
    id: String,
}

impl Key {
    fn new(agent: AgentType, env: &AgentEnv, id: &str) -> Option<Self> {
        let root = match agent {
            AgentType::Codex => codex::resolve_codex_home_dir_for_env(env).join("sessions"),
            AgentType::ClaudeCode if !id.is_empty() && super::is_safe_subagent_id(id) => {
                claude::resolve_claude_projects_dir_for_env(env)
            }
            _ => return None,
        };
        root.is_absolute().then(|| Self {
            agent,
            root,
            id: id.to_owned(),
        })
    }
}

struct Parsed {
    detail: Arc<ConversationDetail>,
    sources: sources::Version,
}

type Flight = OnceLock<Option<Arc<Parsed>>>;

struct Entry {
    parsed: Arc<Parsed>,
    bytes: u64,
    last_used: Instant,
}

#[derive(Default)]
struct Store {
    ready: BTreeMap<Key, Entry>,
    busy: BTreeMap<Key, Arc<Flight>>,
    bytes: u64,
}

struct DetailCache {
    budget: u64,
    store: Mutex<Store>,
}

struct Lease<'a> {
    cache: &'a DetailCache,
    key: Key,
    flight: Arc<Flight>,
    result: Option<Arc<Parsed>>,
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        let mut store = self.cache.lock();
        store.busy.remove(&self.key);
        // Also release waiters when the loader fails or unwinds.
        let _ = self.flight.set(self.result.take());
    }
}

impl DetailCache {
    fn new(budget: u64) -> Self {
        Self {
            budget,
            store: Mutex::new(Store::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Store> {
        self.store.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn get(
        &self,
        key: Key,
        parse: impl FnOnce() -> Result<ConversationDetail, ParseError>,
    ) -> Result<Arc<ConversationDetail>, ParseError> {
        let (mut lease, cached) = loop {
            let mut store = self.lock();
            if let Some(pending) = store.busy.get(&key).cloned() {
                drop(store);
                let Some(parsed) = pending.wait() else {
                    // An unstable source, an error or an interrupted read has no
                    // snapshot to share. Each waiter can now read independently.
                    return parse().map(Arc::new);
                };
                if parsed.sources.current(&key) {
                    return Ok(parsed.detail.clone());
                }
                continue;
            }
            let flight = Arc::new(OnceLock::new());
            store.busy.insert(key.clone(), flight.clone());
            let cached = store.ready.get_mut(&key).map(|entry| {
                entry.last_used = Instant::now();
                entry.parsed.clone()
            });
            break (
                Lease {
                    key,
                    cache: self,
                    flight,
                    result: None,
                },
                cached,
            );
        };
        let key = &lease.key;
        if let Some(parsed) = cached {
            if parsed.sources.current(key) {
                let detail = parsed.detail.clone();
                lease.result = Some(parsed);
                return Ok(detail);
            }
            let mut store = self.lock();
            if let Some(stale) = store.ready.remove(key) {
                store.bytes -= stale.bytes;
                drop(store);
                drop(stale);
            }
        }

        let (detail, version) = sources::record(parse);
        let mut detail = detail?;
        let Some(version) = version.filter(|v| v.current(key)) else {
            return Ok(Arc::new(detail));
        };
        // Charge a full B-tree node per entry (up to eleven slots), including
        // child links, plus allocation headers for the root, id and Arcs.
        let entry_bytes = 12 * (size_of::<(Key, Entry)>() + size_of::<usize>()) + 128;
        let bytes = weight::detail_bytes(&mut detail)
            + version.bytes()
            + key.root.capacity() as u64
            + key.id.capacity() as u64
            + entry_bytes as u64;
        let detail = Arc::new(detail);
        let parsed = Arc::new(Parsed {
            sources: version,
            detail: detail.clone(),
        });
        if bytes <= self.budget {
            let mut evicted = Vec::new();
            let mut store = self.lock();
            if store.bytes > self.budget - bytes {
                // Rank once: admitting one large history can displace thousands of
                // tiny histories. Repeatedly scanning for the oldest is quadratic.
                let mut by_age: Vec<_> = store
                    .ready
                    .iter()
                    .map(|(key, entry)| (entry.last_used, key.clone()))
                    .collect();
                by_age.sort_unstable_by_key(|(used, _)| *used);
                for (_, oldest) in by_age {
                    let removed = store.ready.remove(&oldest).unwrap();
                    store.bytes -= removed.bytes;
                    evicted.push(removed);
                    if store.bytes <= self.budget - bytes {
                        break;
                    }
                }
            }
            store.bytes += bytes;
            store.ready.insert(
                key.clone(),
                Entry {
                    parsed: parsed.clone(),
                    bytes,
                    last_used: Instant::now(),
                },
            );
            drop(store);
            drop(evicted);
        }
        // Existing waiters may share a stable oversized result without retaining
        // it in the bounded store for subsequent requests.
        lease.result = Some(parsed);
        Ok(detail)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[test]
fn unversioned_waiters_parse_in_parallel_after_the_leader_finishes() {
    use std::fs;
    use std::sync::{mpsc, Condvar};
    use std::time::Duration;

    let home = tempfile::tempdir().unwrap();
    let project = home.path().join("projects/demo");
    fs::create_dir_all(&project).unwrap();
    // Missing timestamps force the parser to consult the wall clock.
    fs::write(
        project.join("active.jsonl"),
        r#"{"type":"user","message":{"role":"user","content":"hello"}}"#,
    )
    .unwrap();
    let env = AgentEnv::with_overrides(&[("CLAUDE_CONFIG_DIR", home.path().to_str().unwrap())]);
    let key = Key::new(AgentType::ClaudeCode, &env, "active").unwrap();
    let cache = DetailCache::new(CACHE_BYTES);
    let parse = || build_agent_parser(key.agent, &env).get_conversation("active");
    let timeout = Duration::from_secs(5);
    let (started, entered) = mpsc::channel();
    let followers = (Mutex::new(0), Condvar::new());

    std::thread::scope(|scope| {
        scope.spawn(|| {
            cache.get(key.clone(), || {
                started.send(()).unwrap();
                let deadline = Instant::now() + timeout;
                // Map, leader and two waiting readers each own the flight.
                while Arc::strong_count(cache.lock().busy.get(&key).unwrap()) < 4 {
                    assert!(Instant::now() < deadline, "readers did not join the flight");
                    std::thread::yield_now();
                }
                parse()
            })
            .unwrap()
        });
        entered.recv_timeout(timeout).unwrap();
        for _ in 0..2 {
            scope.spawn(|| {
                cache.get(key.clone(), || {
                    let mut count = followers.0.lock().unwrap();
                    *count += 1;
                    followers.1.notify_all();
                    let (count, _) = followers
                        .1
                        .wait_timeout_while(count, timeout, |count| *count < 2)
                        .unwrap();
                    let simultaneous = *count == 2;
                    drop(count);
                    assert!(simultaneous, "unversioned reads were serialized");
                    parse()
                })
                .unwrap()
            });
        }
    });
    assert!(cache.lock().ready.is_empty());
}
