// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-choice isolation regression lane for the v1 jail (DIS-2381 step 3).
//!
//! Invariant under test:
//!
//! ```text
//!   demux(parse(interleave(A@0, B@1))) == (parse(A), parse(B))
//! ```
//!
//! `JailedStream` keys its jail/marker/tool-call state off `choice.index` via
//! `ChoiceJailStateCollection`. A regression that shared one state across indices
//! (or routed every delta to index 0) would still pass every existing test,
//! because all existing jail tests are effectively single-choice: even
//! `test_multiple_choices_independent_jailing` delivers each choice's deltas in
//! its own slot of a packed multi-choice chunk, so a shared accumulator that
//! processed choices in a fixed order could still look correct.
//!
//! This lane instead runs two independent choices through ONE `JailedStream` with
//! their deltas *interleaved on the wire* under several deterministic schedules
//! (round-robin, first-byte offset, mid-delta boundary split — see
//! `common/interleave.rs`). It then demuxes the emitted chunks by `choice.index`
//! and asserts each choice's assembled result (tool calls, normal text, finish
//! handling) is byte-for-byte what that choice produced running ALONE. If jail
//! state leaks across choices, one choice's partial marker/JSON corrupts the
//! other and the demuxed assembly diverges from its solo golden.
//!
//! Golden = the solo run (never a hand-authored n>1 expectation). Because the
//! jail reassembles content across arbitrary delta boundaries, the assembled
//! result is invariant to where a delta is split, so the same solo golden is
//! valid for every schedule.

#[path = "common/interleave.rs"]
mod interleave;

use dynamo_parsers::tool_calling::jail::{Annotated, JailedStream};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, FinishReason, Role,
};
use futures::{StreamExt, stream};
use interleave::{Schedule, interleave_items};
use std::collections::BTreeMap;

/// Build a single-choice content chunk tagged with `index` (mirrors the shape of
/// `create_mock_response_chunk` in `jail.rs`).
fn single_choice_chunk(content: &str, index: u32) -> Annotated<CreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: Some(ChatCompletionMessageContent::Text(content.to_string())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    };
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: "jail-interleave".to_string(),
            choices: vec![choice],
            created: 0,
            model: "test-model".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

/// Assembled per-choice view of jail output: the parts a real n>1 client sees.
#[derive(Debug, PartialEq)]
struct Assembled {
    /// `(name, arguments)` per tool call, accumulated by tool-call index.
    tool_calls: Vec<(String, String)>,
    /// Concatenated non-tool-call content.
    normal_text: String,
    /// Terminal finish reason the jail attributed to this choice, if any.
    finish: Option<FinishReason>,
}

/// Assemble one choice's emitted chunks (already demuxed by `choice.index`).
fn assemble(chunks: &[&ChatChoiceStream]) -> Assembled {
    let mut normal_text = String::new();
    let mut names: BTreeMap<u32, String> = BTreeMap::new();
    let mut args: BTreeMap<u32, String> = BTreeMap::new();
    let mut order: Vec<u32> = Vec::new();
    let mut finish: Option<FinishReason> = None;

    for choice in chunks {
        if let Some(ChatCompletionMessageContent::Text(text)) = choice.delta.content.as_ref() {
            normal_text.push_str(text);
        }
        if let Some(calls) = choice.delta.tool_calls.as_ref() {
            for call in calls {
                if !order.contains(&call.index) {
                    order.push(call.index);
                }
                if let Some(function) = call.function.as_ref() {
                    if let Some(name) = function.name.as_ref() {
                        names.entry(call.index).or_default().push_str(name);
                    }
                    if let Some(a) = function.arguments.as_ref() {
                        args.entry(call.index).or_default().push_str(a);
                    }
                }
            }
        }
        if let Some(reason) = choice.finish_reason {
            finish = Some(reason);
        }
    }

    let tool_calls = order
        .into_iter()
        .map(|idx| {
            (
                names.get(&idx).cloned().unwrap_or_default(),
                args.get(&idx).cloned().unwrap_or_default(),
            )
        })
        .collect();

    Assembled {
        tool_calls,
        normal_text,
        finish,
    }
}

/// Run one choice's deltas solo through a fresh `JailedStream` and assemble the
/// single-choice output. This is the golden the interleaved run must reproduce.
async fn solo(parser: &str, deltas: &[String]) -> Assembled {
    let chunks: Vec<_> = deltas.iter().map(|d| single_choice_chunk(d, 0)).collect();
    let results: Vec<_> = JailedStream::builder()
        .tool_call_parser(parser)
        .build()
        .apply_with_finish_reason(stream::iter(chunks))
        .collect()
        .await;
    let choices: Vec<&ChatChoiceStream> = results
        .iter()
        .filter_map(|r| r.data.as_ref())
        .flat_map(|d| d.choices.iter())
        .collect();
    assemble(&choices)
}

/// Feed an interleaved multi-choice stream through ONE `JailedStream`, then demux
/// the emitted chunks by `choice.index` and assemble each choice separately.
async fn interleaved_by_choice(
    parser: &str,
    sequences: &[Vec<String>],
    schedule: Schedule,
) -> BTreeMap<u32, Assembled> {
    let tagged = interleave_items(sequences, schedule);
    let chunks: Vec<_> = tagged
        .iter()
        .map(|(index, content)| single_choice_chunk(content, *index))
        .collect();
    let results: Vec<_> = JailedStream::builder()
        .tool_call_parser(parser)
        .build()
        .apply_with_finish_reason(stream::iter(chunks))
        .collect()
        .await;

    // Demux by `choice.index` (NEVER by arrival order): flat_map every choice out
    // of every emitted chunk — the jail may pack or split, so we must not assume
    // one choice per chunk.
    let mut per_choice: BTreeMap<u32, Vec<&ChatChoiceStream>> = BTreeMap::new();
    for choice in results
        .iter()
        .filter_map(|r| r.data.as_ref())
        .flat_map(|d| d.choices.iter())
    {
        per_choice.entry(choice.index).or_default().push(choice);
    }
    per_choice
        .into_iter()
        .map(|(index, chunks)| (index, assemble(&chunks)))
        .collect()
}

/// A named divergent pair: two choices that produce structurally different output
/// through the same parser, so a shared accumulator visibly corrupts one.
struct Pair {
    name: &'static str,
    parser: &'static str,
    sequences: Vec<Vec<String>>,
}

fn s(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

fn divergent_pairs() -> Vec<Pair> {
    vec![
        // (tool call) x (plain content only): the classic n>1 leak — a jailed
        // tool call in choice 0 must not swallow choice 1's plain prose.
        Pair {
            name: "hermes_toolcall_x_plain",
            parser: "hermes",
            sequences: vec![
                s(&[
                    "Let me check. ",
                    r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#,
                    " done.",
                ]),
                s(&["Just ", "plain ", "prose, no tools here."]),
            ],
        },
        // (two different tool-call shapes): both choices emit tool calls but with
        // different names/args — a shared buffer would cross-contaminate arguments.
        Pair {
            name: "hermes_two_distinct_calls",
            parser: "hermes",
            sequences: vec![
                s(&[
                    r#"<tool_call>{"name":"get_weather","#,
                    r#""arguments":{"city":"Paris"}}</tool_call>"#,
                ]),
                s(&[
                    r#"<tool_call>{"name":"get_time","#,
                    r#""arguments":{"tz":"UTC"}}</tool_call>"#,
                ]),
            ],
        },
        // (opening-marker-split-across-deltas) x (bare content): choice 0's
        // `<tool_call>` marker is split across delta boundaries; a shared partial
        // marker buffer would jail choice 1's bare content by mistake.
        Pair {
            name: "hermes_split_marker_x_bare",
            parser: "hermes",
            sequences: vec![
                s(&[
                    "<tool",
                    "_call>",
                    r#"{"name":"lookup","arguments":{"q":"cats"}}"#,
                    "</tool_call>",
                ]),
                s(&["bare content only, ", "never jailed"]),
            ],
        },
    ]
}

fn schedules_k2() -> Vec<Schedule> {
    vec![
        Schedule::RoundRobin,
        Schedule::FirstByteOffset(1),
        Schedule::FirstByteOffset(2),
        Schedule::BoundarySplit,
    ]
}

/// Whether a schedule can be meaningfully applied to `sequences`. Un-applicable
/// shapes are logged as skipped rather than silently counted as passing.
fn applicable(sequences: &[Vec<String>], schedule: Schedule) -> Result<(), String> {
    if sequences.iter().any(|s| s.is_empty()) {
        return Err("a choice has no deltas".to_string());
    }
    match schedule {
        Schedule::BoundarySplit => {
            if sequences.len() != 2 {
                return Err("BoundarySplit requires exactly two choices".to_string());
            }
            if sequences[0].iter().all(|d| d.chars().count() < 2) {
                return Err("choice 0 has no splittable delta".to_string());
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[tokio::test]
async fn jail_interleave_preserves_per_choice_isolation() {
    let mut ran = 0usize;
    let mut skipped = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for pair in divergent_pairs() {
        assert_eq!(pair.sequences.len(), 2, "k=2 pairs only in this loop");
        // Solo goldens (one fresh JailedStream per choice).
        let golden_a = solo(pair.parser, &pair.sequences[0]).await;
        let golden_b = solo(pair.parser, &pair.sequences[1]).await;

        for schedule in schedules_k2() {
            if let Err(reason) = applicable(&pair.sequences, schedule) {
                eprintln!(
                    "SKIP pair={} schedule={}: {reason}",
                    pair.name,
                    schedule.label()
                );
                skipped += 1;
                continue;
            }
            ran += 1;

            let demuxed = interleaved_by_choice(pair.parser, &pair.sequences, schedule).await;
            for (index, golden) in [(0u32, &golden_a), (1u32, &golden_b)] {
                match demuxed.get(&index) {
                    Some(got) if got == golden => {}
                    Some(got) => failures.push(format!(
                        "schedule={} pair={} choice={index} diverged:\n     got  {got:?}\n     want {golden:?}",
                        schedule.label(),
                        pair.name,
                    )),
                    None => failures.push(format!(
                        "schedule={} pair={} choice={index}: no output demuxed for this choice",
                        schedule.label(),
                        pair.name,
                    )),
                }
            }
        }
    }

    eprintln!("v1 jail interleave: {ran} schedule-pairs ran, {skipped} skipped");
    assert!(ran >= 8, "expected the divergent-pair matrix to run");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// One k=3 round-robin case: three choices (tool call / plain / tool call) share
/// one `JailedStream`; each must demux to its own solo golden.
#[tokio::test]
async fn jail_interleave_three_choices_round_robin() {
    let parser = "hermes";
    let sequences = vec![
        s(&[
            r#"<tool_call>{"name":"a_call","arguments":{"x":1}}</tool_call>"#,
            " after a",
        ]),
        s(&["plain ", "middle ", "content"]),
        s(&[
            "prefix ",
            r#"<tool_call>{"name":"c_call","arguments":{"y":2}}</tool_call>"#,
        ]),
    ];

    let goldens: Vec<Assembled> = {
        let mut v = Vec::new();
        for seq in &sequences {
            v.push(solo(parser, seq).await);
        }
        v
    };

    let demuxed = interleaved_by_choice(parser, &sequences, Schedule::RoundRobin).await;
    let mut failures = Vec::new();
    for (index, golden) in goldens.iter().enumerate() {
        let index = index as u32;
        match demuxed.get(&index) {
            Some(got) if got == golden => {}
            Some(got) => failures.push(format!(
                "k=3 RoundRobin choice={index} diverged:\n     got  {got:?}\n     want {golden:?}"
            )),
            None => failures.push(format!("k=3 RoundRobin choice={index}: no output demuxed")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
