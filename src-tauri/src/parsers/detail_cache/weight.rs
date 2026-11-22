//! Admission measures owned allocations, including unused String/Vec capacity.
//! Inline fields are already paid for by their containing object or vector.
//! Closed matches on ContentBlock make a new wire variant a compile-time check.

use crate::models::{AgentExecutionStats, ContentBlock, ConversationDetail, ImageData};
use serde_json::Value;
use std::mem::size_of;

fn allocation(bytes: usize) -> u64 {
    if bytes == 0 {
        0
    } else {
        bytes as u64 + 32
    }
}

fn string(s: &String) -> u64 {
    allocation(s.capacity())
}

fn optional(s: &Option<String>) -> u64 {
    s.as_ref().map_or(0, string)
}

fn vec_storage<T>(v: &Vec<T>) -> u64 {
    allocation(v.capacity() * size_of::<T>())
}

fn image(i: &ImageData) -> u64 {
    string(&i.data) + string(&i.mime_type) + optional(&i.uri)
}

fn agent(s: &AgentExecutionStats) -> u64 {
    optional(&s.agent_type)
        + optional(&s.status)
        + optional(&s.recovered_result)
        + optional(&s.child_session_id)
        + vec_storage(&s.tool_calls)
        + s.tool_calls
            .iter()
            .map(|c| {
                string(&c.tool_name) + optional(&c.input_preview) + optional(&c.output_preview)
            })
            .sum::<u64>()
        + vec_storage(&s.timeline)
        + s.timeline
            .iter()
            .map(|t| string(&t.entry_type) + optional(&t.text))
            .sum::<u64>()
}

fn json_heap(value: &mut Value) -> u64 {
    match value {
        Value::String(s) => string(s),
        Value::Array(v) => vec_storage(v) + v.iter_mut().map(json_heap).sum::<u64>(),
        Value::Object(map) => {
            // serde_json can use a BTreeMap or IndexMap (unified Cargo features).
            // Neither exposes capacity here. Rebuild with the final length so
            // deleted keys cannot leave an unmeasurably large retained table.
            let len = map.len();
            let old = std::mem::replace(map, serde_json::Map::with_capacity(len));
            let mut nested = 0;
            for (key, mut value) in old {
                nested += string(&key) + json_heap(&mut value);
                map.insert(key, value);
            }
            // Allow twelve key/value slots and child links per B-tree node;
            // a non-root node has at least five keys. This also covers a
            // pre-sized IndexMap's entries and hash buckets.
            let nodes = if len == 0 { 0 } else { 1 + len / 5 };
            nested
                + nodes as u64
                    * (64 + 12 * (size_of::<(String, Value)>() + size_of::<usize>())) as u64
        }
        Value::Number(n) if size_of::<serde_json::Number>() >= size_of::<String>() => {
            // With arbitrary_precision enabled Number owns its decimal string.
            // Cloning removes any spare capacity in the hidden backing String.
            *n = n.clone();
            allocation(n.to_string().len())
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => 0,
    }
}

fn block_heap(block: &mut ContentBlock) -> u64 {
    match block {
        ContentBlock::Text { text } | ContentBlock::Thinking { text } => string(text),
        ContentBlock::TurnError { message } => string(message),
        ContentBlock::Image {
            data,
            mime_type,
            uri,
        } => string(data) + string(mime_type) + optional(uri),
        ContentBlock::ImageGeneration {
            revised_prompt,
            image: img,
        } => optional(revised_prompt) + img.as_ref().map_or(0, image),
        ContentBlock::ToolUse {
            tool_use_id,
            tool_name,
            input_preview,
            status,
            meta,
        } => {
            optional(tool_use_id)
                + string(tool_name)
                + optional(input_preview)
                + optional(status)
                + meta.as_mut().map_or(0, json_heap)
        }
        ContentBlock::ToolResult {
            tool_use_id,
            output_preview,
            agent_stats,
            images,
            ..
        } => {
            optional(tool_use_id)
                + optional(output_preview)
                + agent_stats.as_ref().map_or(0, agent)
                + vec_storage(images)
                + images.iter().map(image).sum::<u64>()
        }
    }
}

pub(super) fn detail_bytes(detail: &mut ConversationDetail) -> u64 {
    let s = &detail.summary;
    let mut bytes = size_of::<ConversationDetail>() as u64
        + string(&s.id)
        + optional(&s.folder_path)
        + optional(&s.folder_name)
        + optional(&s.title)
        + optional(&s.model)
        + optional(&s.git_branch)
        + optional(&s.parent_id)
        + optional(&s.parent_tool_use_id)
        + optional(&s.delegation_call_id)
        + vec_storage(&detail.turns);
    for turn in &mut detail.turns {
        bytes += string(&turn.id)
            + optional(&turn.model)
            + optional(&turn.agent_message_id)
            + vec_storage(&turn.blocks)
            + turn.blocks.iter_mut().map(block_heap).sum::<u64>();
    }
    bytes
}
