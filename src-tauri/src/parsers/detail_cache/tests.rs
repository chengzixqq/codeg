use super::*;
use crate::acp::agent_mentions::append_agent_routes;
use crate::acp::types::PromptInputBlock;
use crate::models::ContentBlock;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Barrier;

const TS: &str = "2026-09-01T10:00:00Z";

struct History {
    home: tempfile::TempDir,
    agent: AgentType,
    env: AgentEnv,
}

impl History {
    fn new(agent: AgentType) -> Self {
        let home = tempfile::tempdir().unwrap();
        let var = match agent {
            AgentType::Codex => "CODEX_HOME",
            AgentType::ClaudeCode => "CLAUDE_CONFIG_DIR",
            _ => unreachable!(),
        };
        let env = AgentEnv::with_overrides(&[(var, home.path().to_str().unwrap())]);
        Self { home, agent, env }
    }

    fn claude(&self, id: &str) -> PathBuf {
        self.home
            .path()
            .join("projects/demo")
            .join(format!("{id}.jsonl"))
    }

    fn codex(&self, date: &str, id: &str) -> PathBuf {
        self.home
            .path()
            .join("sessions")
            .join(date)
            .join(format!("rollout-{date}T10-00-00-{id}.jsonl"))
    }

    fn get(&self, id: &str) -> Arc<ConversationDetail> {
        shared_conversation_detail(self.agent, &self.env, id).unwrap()
    }
}

fn write(path: &Path, records: &[Value]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        path,
        records
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
}

fn user(text: &str) -> Value {
    json!({"type":"user","timestamp":TS,"uuid":"u","message":{"role":"user","content":text}})
}

fn assistant(text: &str, tokens: u64) -> Value {
    json!({"type":"assistant","timestamp":TS,"uuid":"a","message":{"id":"a","role":"assistant","content":[{"type":"text","text":text}],"usage":{"input_tokens":tokens}}})
}

fn record(kind: &str, payload: Value) -> Value {
    json!({"type":kind, "timestamp":TS, "payload":payload})
}

fn text_in(detail: &ConversationDetail, expected: &str) -> bool {
    detail
        .turns
        .iter()
        .flat_map(|t| &t.blocks)
        .any(|b| matches!(b, ContentBlock::Text { text } if text == expected))
}

#[test]
fn unchanged_claude_detail_shares_the_sanitized_parse_and_isolates_agent_roots() {
    let a = History::new(AgentType::ClaudeCode);
    let b = History::new(AgentType::ClaudeCode);
    let mut blocks = vec![PromptInputBlock::Text {
        text: "ask [@A](codeg://agent/codex) to help".into(),
    }];
    append_agent_routes(&mut blocks, true);
    let PromptInputBlock::Text { text: frame } = &blocks[1] else {
        unreachable!()
    };
    write(&a.claude("s"), &[user(&format!("hello\n{frame}"))]);
    write(&b.claude("s"), &[user("second")]);
    let first = a.get("s");
    assert!(text_in(&first, "hello"));
    assert!(Arc::ptr_eq(&first, &a.get("s")));
    assert!(text_in(&b.get("s"), "second"));
}

#[test]
fn claude_sees_atomic_replacement_with_preserved_size_and_mtime() {
    let h = History::new(AgentType::ClaudeCode);
    let path = h.claude("s");
    write(&path, &[user("before")]);
    let old = h.get("s");
    let temp = path.with_extension("tmp");
    write(&temp, &[user("after!")]);
    let mtime = fs::metadata(&path).unwrap().modified().unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(&temp)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
    fs::rename(&temp, &path).unwrap();
    assert!(text_in(&old, "before"));
    assert!(text_in(&h.get("s"), "after!"));
}

#[test]
fn claude_keeps_the_parent_parse_while_other_project_sessions_write() {
    use std::io::Write;
    let h = History::new(AgentType::ClaudeCode);
    let parent = h.claude("parent");
    write(
        &parent,
        &[user(&"long history ".repeat(8192)), assistant("done", 5)],
    );
    let child = parent
        .with_extension("")
        .join("subagents/agent-worker.jsonl");
    write(&child, &[assistant("worker", 17)]);
    let sibling = h.claude("parallel-session");
    write(&sibling, &[user("independent conversation")]);
    let before = h.get("parent");
    assert!(text_in(&before, "done"));
    for i in 0..3 {
        let mut file = fs::OpenOptions::new().append(true).open(&sibling).unwrap();
        writeln!(file, "\n{}", assistant("still working", i)).unwrap();
        assert!(Arc::ptr_eq(&before, &h.get("parent")), "sibling append {i}");
        write(&h.claude(&format!("other-{i}")), &[user("another session")]);
        assert!(Arc::ptr_eq(&before, &h.get("parent")), "new sibling {i}");
    }
    write(&child, &[assistant("worker", 23)]);
    let updated = h.get("parent");
    assert!(!Arc::ptr_eq(&before, &updated));
    assert_eq!(
        updated
            .turns
            .iter()
            .filter_map(|t| t.usage.as_ref())
            .map(|u| u.input_tokens)
            .sum::<u64>(),
        28
    );
}

#[test]
#[cfg(target_os = "linux")]
fn claude_probes_only_the_head_of_a_recent_unrelated_transcript() {
    // Linux accounts read(2) bytes per thread, including reads served by the
    // page cache. A full SHA pass over the sibling alone exceeds this budget.
    fn bytes_read() -> u64 {
        fs::read_to_string("/proc/thread-self/io")
            .unwrap()
            .lines()
            .find_map(|line| line.strip_prefix("rchar: "))
            .unwrap()
            .parse()
            .unwrap()
    }
    let h = History::new(AgentType::ClaudeCode);
    write(&h.claude("parent"), &[user(&"main".repeat(1024))]);
    let sibling = h.claude("busy");
    write(&sibling, &vec![user("unrelated"); 48]);
    fs::File::options()
        .write(true)
        .open(&sibling)
        .unwrap()
        .set_len(4 * 1024 * 1024)
        .unwrap();
    let before = bytes_read();
    let detail = h.get("parent");
    assert_eq!(detail.summary.id, "parent");
    assert!(
        bytes_read() - before < 1024 * 1024,
        "read the unrelated transcript body"
    );

    // Let real ctimes leave the racy-stat window, then prime the settled probes.
    std::thread::sleep(std::time::Duration::from_secs(3));
    assert!(Arc::ptr_eq(&detail, &h.get("parent")));
    let before = bytes_read();
    assert!(Arc::ptr_eq(&detail, &h.get("parent")));
    assert!(bytes_read() - before < 1024, "re-read a settled transcript");
}

#[test]
fn claude_retries_an_incomplete_rollover_query() {
    let h = History::new(AgentType::ClaudeCode);
    write(&h.claude("parent"), &[user("original")]);
    let parent = h.get("parent");
    let next = h.claude("next");
    fs::create_dir(&next).unwrap(); // An entry that cannot be read as JSONL.
    let partial = h.get("parent");
    assert!(text_in(&partial, "original"));
    assert!(!Arc::ptr_eq(&parent, &partial));
    assert!(!Arc::ptr_eq(&partial, &h.get("parent")));
    fs::remove_dir(&next).unwrap();
    write(&next, &[user("<command-name>/clear</command-name>")]);
    assert_eq!(h.get("parent").summary.id, "next");
}

#[test]
fn claude_reselects_a_sibling_rewritten_as_a_clear_with_preserved_mtime() {
    let h = History::new(AgentType::ClaudeCode);
    write(&h.claude("parent"), &[user("parent conversation")]);
    let sibling = h.claude("candidate");
    let records = |cmd| {
        [
            user(&format!("<command-name>/{cmd}</command-name>")),
            assistant("sibling reply", 2),
        ]
    };
    write(&sibling, &records("model"));
    let mtime = fs::metadata(&sibling).unwrap().modified().unwrap();
    let old = h.get("parent");
    write(&sibling, &records("clear"));
    fs::File::options()
        .write(true)
        .open(&sibling)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
    let rolled = h.get("parent");
    assert!(!Arc::ptr_eq(&old, &rolled));
    assert_eq!(rolled.summary.id, "candidate");
    assert!(text_in(&rolled, "sibling reply"));
    assert!(Arc::ptr_eq(&rolled, &h.get("parent")));
}

#[test]
fn claude_child_usage_and_rollover_revalidate_the_parent() {
    let h = History::new(AgentType::ClaudeCode);
    let path = h.claude("parent");
    write(&path, &[assistant("parent", 5)]);
    let total = |d: &ConversationDetail| d.turns[0].usage.as_ref().unwrap().input_tokens;
    assert_eq!(total(&h.get("parent")), 5);
    let child = path.with_extension("").join("subagents/agent-child.jsonl");
    write(&child, &[assistant("child", 17)]);
    let with_child = h.get("parent");
    assert_eq!(total(&with_child), 22);
    assert!(Arc::ptr_eq(&with_child, &h.get("parent")));
    write(&child, &[assistant("child", 19)]);
    assert_eq!(total(&h.get("parent")), 24);
    fs::remove_file(&child).unwrap();
    assert_eq!(total(&h.get("parent")), 5);
    write(
        &h.claude("next"),
        &[
            user("<command-name>/clear</command-name>"),
            assistant("new thread", 2),
        ],
    );
    let rolled = h.get("parent");
    assert_eq!(rolled.summary.id, "next");
    assert!(text_in(&rolled, "new thread"));
}

#[test]
fn codex_inheritance_title_and_new_current_rollout_revalidate() {
    let h = History::new(AgentType::Codex);
    let main = h.codex("2026-09-01", "fork");
    write(
        &main,
        &[record(
            "session_meta",
            json!({"id":"fork","history_base":{"thread_id":"base","end_ordinal_exclusive":5}}),
        )],
    );
    assert!(h.get("fork").turns.is_empty());
    let base = h.codex("2026-08-25", "base");
    let base_records = |text| {
        vec![
            record("session_meta", json!({"id":"base"})),
            record("event_msg", json!({"type":"user_message","message":text})),
        ]
    };
    write(&base, &base_records("inherited"));
    let inherited = h.get("fork");
    assert!(text_in(&inherited, "inherited"));
    assert!(Arc::ptr_eq(&inherited, &h.get("fork")));
    write(&base, &base_records("rewritten"));
    assert!(text_in(&h.get("fork"), "rewritten"));
    let index = h.home.path().join("session_index.jsonl");
    write(&index, &[json!({"id":"fork","thread_name":"renamed"})]);
    assert_eq!(h.get("fork").summary.title.as_deref(), Some("renamed"));
    fs::remove_file(&index).unwrap();
    write(
        &h.codex("2026-09-02", "fork_later"),
        &base_records("after revert"),
    );
    assert!(text_in(&h.get("fork"), "after revert"));
}

#[test]
fn codex_agent_capsule_hits_and_reselects_a_new_preferred_child() {
    let h = History::new(AgentType::Codex);
    let main = h.codex("2026-09-01", "parent");
    write(
        &main,
        &[
            record("session_meta", json!({"id":"parent","cwd":"/tmp/demo"})),
            record(
                "response_item",
                json!({"type":"function_call","call_id":"spawn","name":"spawn_agent","arguments":"{\"agent_type\":\"worker\",\"message\":\"do it\"}"}),
            ),
            record(
                "response_item",
                json!({"type":"function_call_output","call_id":"spawn","output":"{\"agent_id\":\"child\"}"}),
            ),
        ],
    );
    let tools = |d: &ConversationDetail| {
        d.turns
            .iter()
            .flat_map(|t| &t.blocks)
            .filter_map(|b| {
                if let ContentBlock::ToolResult {
                    agent_stats: Some(stats),
                    ..
                } = b
                {
                    Some(&stats.tool_calls)
                } else {
                    None
                }
            })
            .flatten()
            .map(|t| t.tool_name.clone())
            .collect::<Vec<_>>()
    };
    let child = |name| {
        record(
            "response_item",
            json!({"type":"function_call","call_id":"local","name":name,"arguments":"{}"}),
        )
    };
    assert!(tools(&h.get("parent")).is_empty());
    write(
        &main.with_file_name("rollout-2026-09-01T09-00-00-child.jsonl"),
        &[child("read_file")],
    );
    let detail = h.get("parent");
    assert_eq!(tools(&detail), ["read_file"]);
    assert!(Arc::ptr_eq(&detail, &h.get("parent")));
    write(
        &main.with_file_name("agent-child.jsonl"),
        &[child("edit_file")],
    );
    assert_eq!(tools(&h.get("parent")), ["edit_file"]);
}

#[test]
fn simultaneous_real_reads_share_the_same_parsed_detail() {
    let h = History::new(AgentType::ClaudeCode);
    write(&h.claude("s"), &[user(&"x".repeat(128 * 1024))]);
    let start = Barrier::new(8);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    start.wait();
                    h.get("s")
                })
            })
            .collect();
        let results: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
        assert!(results.iter().all(|r| Arc::ptr_eq(r, &results[0])));
    });
}

// Exercise admission and coordination on an isolated cache using the same parser
// as production. The injected closure is the expensive, full parse operation.
fn cached<F>(
    cache: &DetailCache,
    history: &History,
    id: &str,
    after: F,
) -> Result<Arc<ConversationDetail>, ParseError>
where
    F: FnOnce(&mut ConversationDetail),
{
    let key = Key::new(history.agent, &history.env, id).unwrap();
    cache.get(key, || {
        let mut detail = build_agent_parser(history.agent, &history.env).get_conversation(id)?;
        after(&mut detail);
        Ok(detail)
    })
}

#[test]
fn independent_keys_parse_in_parallel_and_same_key_runs_the_loader_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let cache = DetailCache::new(1024 * 1024);
    let h = History::new(AgentType::ClaudeCode);
    write(&h.claude("a"), &[user("one")]);
    write(&h.claude("b"), &[user("two")]);
    let completed = (
        Mutex::new(std::collections::BTreeSet::new()),
        std::sync::Condvar::new(),
    );
    let calls = AtomicUsize::new(0);
    std::thread::scope(|s| {
        let handles: Vec<_> = ["a", "a", "b", "b"]
            .into_iter()
            .map(|id| {
                s.spawn({
                    let (h, cache, completed, calls) = (&h, &cache, &completed, &calls);
                    move || {
                        cached(cache, h, id, |_| {
                            calls.fetch_add(1, Ordering::SeqCst);
                            let mut keys = completed.0.lock().unwrap();
                            keys.insert(id);
                            completed.1.notify_all();
                            let (keys, _) = completed
                                .1
                                .wait_timeout_while(
                                    keys,
                                    std::time::Duration::from_secs(5),
                                    |keys| keys.len() < 2,
                                )
                                .unwrap();
                            assert_eq!(keys.len(), 2, "independent parses serialized");
                        })
                        .unwrap()
                    }
                })
            })
            .collect();
        let parsed: Vec<_> = handles.into_iter().map(|w| w.join().unwrap()).collect();
        assert!(Arc::ptr_eq(&parsed[0], &parsed[1]));
        assert!(Arc::ptr_eq(&parsed[2], &parsed[3]));
    });
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn changed_during_parse_failure_and_panic_allow_a_later_stable_hit() {
    let h = History::new(AgentType::ClaudeCode);
    let cache = DetailCache::new(1024 * 1024);
    let path = h.claude("s");
    assert!(cached(&cache, &h, "s", |_| {}).is_err());
    write(&path, &[user("old")]);
    let old = cached(&cache, &h, "s", |_| write(&path, &[user("new")])).unwrap();
    assert!(text_in(&old, "old"));
    assert!(
        std::panic::catch_unwind(|| cached(&cache, &h, "s", |_| panic!("interrupted loader")))
            .is_err()
    );
    let new = cached(&cache, &h, "s", |_| {}).unwrap();
    assert!(text_in(&new, "new"));
    assert!(Arc::ptr_eq(
        &new,
        &cached(&cache, &h, "s", |_| panic!("warm parse")).unwrap()
    ));
}

#[test]
fn nested_capacity_controls_admission_and_least_recently_used_entry_is_evicted() {
    let h = History::new(AgentType::ClaudeCode);
    let cache = DetailCache::new(128 * 1024);
    for id in ["a", "b", "c"] {
        write(&h.claude(id), &[user("hi")]);
    }
    // Reserve slack behind a one-character tool image. Serializing the detail
    // cannot see these bytes, but retaining it would exceed the entire cache.
    let oversized = cached(&cache, &h, "a", |detail| {
        use crate::models::ImageData;
        let mut data = String::with_capacity(256 * 1024);
        data.push('x');
        detail.turns[0].blocks.push(ContentBlock::ImageGeneration {
            revised_prompt: None,
            image: Some(ImageData {
                data,
                mime_type: "image/png".into(),
                uri: None,
            }),
        });
    })
    .unwrap();
    let load = |id| {
        cached(&cache, &h, id, |detail| {
            detail.turns[0].blocks.push(ContentBlock::Text {
                text: String::with_capacity(48 * 1024),
            });
        })
        .unwrap()
    };
    let a = load("a");
    assert!(!Arc::ptr_eq(&oversized, &a));
    let b = load("b");
    assert!(Arc::ptr_eq(&a, &load("a")));
    let _c = load("c");
    assert!(Arc::ptr_eq(&a, &load("a")));
    assert!(!Arc::ptr_eq(&b, &load("b")));
}

#[test]
fn codex_patch_line_recovery_observes_a_later_workspace_file() {
    let h = History::new(AgentType::Codex);
    let target = h.home.path().join("patched.txt");
    let patch = format!(
        "*** Begin Patch\n*** Update File: {}\n@@\n keep\n-old\n+new\n*** End Patch",
        target.display()
    );
    write(
        &h.codex("2026-09-01", "edit"),
        &[
            record("session_meta", json!({"id":"edit"})),
            record(
                "response_item",
                json!({"type":"custom_tool_call","call_id":"call","name":"apply_patch","input":patch}),
            ),
        ],
    );
    let preview = |d: &ConversationDetail| {
        d.turns
            .iter()
            .flat_map(|t| &t.blocks)
            .find_map(|b| {
                if let ContentBlock::ToolUse { input_preview, .. } = b {
                    input_preview.clone()
                } else {
                    None
                }
            })
            .unwrap()
    };
    let before = h.get("edit");
    assert!(preview(&before).contains("@@\n"));
    fs::write(&target, "zero\nkeep\nold\n").unwrap();
    let changed = h.get("edit");
    assert!(preview(&changed).contains("@@ -2,"));
    assert!(Arc::ptr_eq(&changed, &h.get("edit")));
}

#[test]
fn a_wall_clock_fallback_bypasses_until_the_transcript_has_timestamps() {
    let h = History::new(AgentType::ClaudeCode);
    let path = h.claude("s");
    let mut record = user("hello");
    record.as_object_mut().unwrap().remove("timestamp");
    write(&path, &[record]);
    assert!(!Arc::ptr_eq(&h.get("s"), &h.get("s")));
    write(&path, &[user("hello")]);
    assert!(Arc::ptr_eq(&h.get("s"), &h.get("s")));
}

#[test]
fn concurrent_stable_oversized_detail_is_shared_only_with_current_readers() {
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    let cache = DetailCache::new(1);
    let h = History::new(AgentType::ClaudeCode);
    write(&h.claude("large"), &[user("large detail")]);
    let key = Key::new(h.agent, &h.env, "large").unwrap();
    let (entered, start) = mpsc::channel();
    std::thread::scope(|s| {
        let leader = s.spawn(|| {
            cached(&cache, &h, "large", |_| {
                entered.send(()).unwrap();
                let until = Instant::now() + Duration::from_secs(5);
                // One owner in the map, one in the loader, one in the waiting call.
                while Arc::strong_count(cache.lock().busy.get(&key).unwrap()) < 3 {
                    assert!(
                        Instant::now() < until,
                        "reader never joined the in-flight parse"
                    );
                    std::thread::yield_now();
                }
            })
            .unwrap()
        });
        start.recv_timeout(Duration::from_secs(5)).unwrap();
        let follower = s.spawn(|| {
            cached(&cache, &h, "large", |_| {
                panic!("duplicate concurrent parse")
            })
            .unwrap()
        });
        let a = leader.join().unwrap();
        assert!(Arc::ptr_eq(&a, &follower.join().unwrap()));
        assert!(!Arc::ptr_eq(
            &a,
            &cached(&cache, &h, "large", |_| {}).unwrap()
        ));
    });
}

#[test]
fn claude_append_truncate_remove_and_relocate_refresh_turns_and_watermark() {
    use std::io::Write;
    let h = History::new(AgentType::ClaudeCode);
    let path = h.claude("s");
    write(&path, &[user("first")]);
    let first = h.get("s");
    let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
    writeln!(file, "\n{}", assistant("reply", 9)).unwrap();
    drop(file);
    let appended = h.get("s");
    assert_eq!(appended.turns.len(), 2);
    assert!(text_in(&appended, "reply"));
    assert_eq!(
        appended.transcript_watermark,
        Some(fs::metadata(&path).unwrap().len())
    );
    write(&path, &[user("short")]);
    let truncated = h.get("s");
    assert_eq!(truncated.turns.len(), 1);
    assert!(text_in(&truncated, "short"));
    fs::remove_file(&path).unwrap();
    assert!(matches!(
        shared_conversation_detail(h.agent, &h.env, "s"),
        Err(ParseError::ConversationNotFound(_))
    ));
    let new_path = h.home.path().join("projects/moved/s.jsonl");
    write(&new_path, &[user("other project")]);
    let moved = h.get("s");
    assert!(text_in(&moved, "other project"));
    assert!(Arc::ptr_eq(&moved, &h.get("s")));
    assert!(text_in(&first, "first"));
}
