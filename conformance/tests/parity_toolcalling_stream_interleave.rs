// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-choice isolation sweep for Dynamo parser v2 (DIS-2381 step 3).
//!
//! Invariant under test, applied across the whole streamv2 corpus:
//!
//! ```text
//!   demux(parse(interleave(A@0, B@1))) == (parse(A), parse(B))
//! ```
//!
//! `parity_toolcalling_stream.rs` proves each case parses correctly on its own.
//! It cannot prove a `ToolParser` keeps its state isolated per `choice.index`,
//! because every fixture is single-choice. When a real `n>1` caller multiplexes
//! several completions onto one wire, it must hand each choice's delta to that
//! choice's own parser instance; a host that shared one parser across choices
//! would splice one choice's partial marker/JSON into another.
//!
//! This lane builds the missing multi-choice stream from the existing corpus: it
//! pairs cases *within a family* (same tools/parser), interleaves them under the
//! deterministic schedules from `parsers/v1/tests/common/interleave.rs` (reused
//! directly — one source of truth across crates), routes each tagged delta to a
//! per-`choice.index` parser via a `ChoiceRouter`, then demuxes and asserts each
//! choice's assembled calls + normal text equal that choice's SOLO run through a
//! fresh single parser. Golden is the solo run, never a hand-authored n>1
//! expectation, so this checks isolation rather than re-checking parity.

#[path = "../../parsers/v1/tests/common/interleave.rs"]
mod interleave;

mod common;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

use common::{collect_yaml, ensure_fixtures};
use dynamo_parsers_v2::{
    Tool, ToolCallDelta, ToolParser, ToolParserInput, create_tool_parser_for_family,
};
use interleave::{Schedule, demux_items, interleave_items};
use serde::Deserialize;
use serde_json::Value;

// ── Family registry (single source of truth) ────────────────────────────────

/// Rows of `conformance/utils/src/parser_families.yaml`. A family is exercised
/// here iff it has a non-null `dynamo_v2` id — derived, never hardcoded, so
/// registering a new v2 family auto-enrolls it.
#[derive(Deserialize)]
struct Registry {
    families: BTreeMap<String, FamilyRow>,
}

#[derive(Deserialize)]
struct FamilyRow {
    dynamo_v2: Option<String>,
    /// `tokens` (Harmony token-native path) or `text`.
    preferred_input: String,
}

fn load_registry() -> Registry {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("utils/src/parser_families.yaml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: read error: {e}", path.display()));
    serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("{}: parse error: {e}", path.display()))
}

// ── Fixture schema (inputs only; goldens are solo runs, not stored expected) ──

#[derive(Deserialize)]
struct Fixture {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize, Clone)]
struct Case {
    #[serde(default)]
    tools: Vec<Tool>,
    #[serde(default)]
    chunks: Vec<Chunk>,
}

#[derive(Deserialize, Clone)]
struct Chunk {
    #[serde(default)]
    delta_text: String,
    #[serde(default)]
    delta_token_ids: Vec<u32>,
}

// ── Assembled per-choice result ──────────────────────────────────────────────

#[derive(Debug, PartialEq)]
struct EngineResult {
    calls: Vec<(String, Value)>,
    normal_text: String,
}

/// Fold a choice's emitted tool-call deltas + normal text into assembled calls.
/// `tool_index` is parser-local; each choice owns its parser so indices never
/// collide across choices.
fn assemble(deltas: &[ToolCallDelta], normal_text: String) -> EngineResult {
    let mut names: BTreeMap<usize, String> = BTreeMap::new();
    let mut args: BTreeMap<usize, String> = BTreeMap::new();
    let mut order: Vec<usize> = Vec::new();
    for d in deltas {
        if !order.contains(&d.tool_index) {
            order.push(d.tool_index);
        }
        if let Some(name) = &d.name {
            names.entry(d.tool_index).or_default().push_str(name);
        }
        args.entry(d.tool_index).or_default().push_str(&d.arguments);
    }
    let calls = order
        .into_iter()
        .map(|idx| {
            let name = names.get(&idx).cloned().unwrap_or_default();
            let raw = args.get(&idx).cloned().unwrap_or_default();
            let v = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            (name, v)
        })
        .collect();
    EngineResult { calls, normal_text }
}

// ── ChoiceRouter: one parser instance per choice.index ───────────────────────

/// Routes tagged deltas to a per-`choice.index` `ToolParser`, exactly what an
/// `n>1` caller must do: a completion's deltas always reach the parser holding
/// that completion's state, never a sibling's. Building one parser per index on
/// first sight (with that index's tools) mirrors real per-request construction.
struct ChoiceRouter {
    family: String,
    parsers: HashMap<u32, Box<dyn ToolParser>>,
    deltas: HashMap<u32, Vec<ToolCallDelta>>,
    normal: HashMap<u32, String>,
}

impl ChoiceRouter {
    fn new(family: &str) -> Self {
        Self {
            family: family.to_string(),
            parsers: HashMap::new(),
            deltas: HashMap::new(),
            normal: HashMap::new(),
        }
    }

    fn push(
        &mut self,
        index: u32,
        tools: &[Tool],
        input: ToolParserInput<'_>,
    ) -> anyhow::Result<()> {
        let res = {
            let family = &self.family;
            let parser = match self.parsers.entry(index) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(create_tool_parser_for_family(family, tools)?)
                }
            };
            parser.push_input(input)?
        };
        self.normal
            .entry(index)
            .or_default()
            .push_str(&res.normal_text);
        self.deltas.entry(index).or_default().extend(res.calls);
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        let indices: Vec<u32> = self.parsers.keys().copied().collect();
        for index in indices {
            let res = self.parsers.get_mut(&index).unwrap().finish()?;
            self.normal
                .entry(index)
                .or_default()
                .push_str(&res.normal_text);
            self.deltas.entry(index).or_default().extend(res.calls);
        }
        Ok(())
    }

    fn assembled(&self, index: u32) -> EngineResult {
        assemble(
            self.deltas.get(&index).map(Vec::as_slice).unwrap_or(&[]),
            self.normal.get(&index).cloned().unwrap_or_default(),
        )
    }
}

/// Run one choice's items solo through a fresh single parser (the golden).
fn solo<T>(
    family: &str,
    tools: &[Tool],
    items: &[T],
    to_input: fn(&T) -> ToolParserInput<'_>,
) -> anyhow::Result<EngineResult> {
    let mut router = ChoiceRouter::new(family);
    for item in items {
        router.push(0, tools, to_input(item))?;
    }
    router.finish()?;
    Ok(router.assembled(0))
}

// ── Generic pair check over one item representation ──────────────────────────

#[allow(clippy::too_many_arguments)]
fn check_pair<T: interleave::Splittable>(
    family: &str,
    tools_a: &[Tool],
    seq_a: &[T],
    tools_b: &[Tool],
    seq_b: &[T],
    schedule: Schedule,
    to_input: fn(&T) -> ToolParserInput<'_>,
    label: &str,
    failures: &mut Vec<String>,
) -> anyhow::Result<()> {
    let sequences = vec![seq_a.to_vec(), seq_b.to_vec()];
    let tagged = interleave_items(&sequences, schedule);
    let demuxed = demux_items(&tagged);
    let tools_by_index = [tools_a, tools_b];

    // Interleaved run through ONE router (per-choice parsers).
    let mut router = ChoiceRouter::new(family);
    for (index, item) in &tagged {
        router.push(*index, tools_by_index[*index as usize], to_input(item))?;
    }
    router.finish()?;

    for index in [0u32, 1] {
        let items = demuxed.get(&index).cloned().unwrap_or_default();
        // Golden = solo run of THIS choice's demuxed subsequence. For splitting
        // schedules the subsequence is finer-grained than the original case but
        // concatenates to identical bytes/tokens; the only variable vs. the
        // interleaved run is the presence of the other choice's deltas.
        let golden = solo(family, tools_by_index[index as usize], &items, to_input)?;
        let got = router.assembled(index);
        if got != golden {
            failures.push(format!(
                "schedule={} {label} choice={index} diverged:\n     got  {got:?}\n     want {golden:?}",
                schedule.label(),
            ));
        }
    }
    Ok(())
}

fn text_input(s: &String) -> ToolParserInput<'_> {
    ToolParserInput::Text(s.as_str())
}

#[allow(clippy::ptr_arg)]
fn token_input(v: &Vec<u32>) -> ToolParserInput<'_> {
    ToolParserInput::Tokens(v.as_slice())
}

// ── Deterministic within-family pairing ──────────────────────────────────────

/// Adjacent pairs + first-with-last over sorted case IDs. Budget-bounded (not
/// all-pairs); deterministic (no RNG).
fn pairs_for(ids: &[String]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..ids.len().saturating_sub(1) {
        out.push((i, i + 1));
    }
    if ids.len() >= 3 {
        out.push((0, ids.len() - 1));
    }
    out
}

fn schedules() -> Vec<Schedule> {
    vec![
        Schedule::RoundRobin,
        Schedule::FirstByteOffset(1),
        Schedule::FirstByteOffset(2),
        Schedule::BoundarySplit,
    ]
}

// ── Test ─────────────────────────────────────────────────────────────────────

#[test]
fn toolcalling_stream_interleave_isolation() {
    let started = Instant::now();
    let registry = load_registry();
    let enabled: BTreeMap<&str, &FamilyRow> = registry
        .families
        .iter()
        .filter(|(_, row)| row.dynamo_v2.is_some())
        .map(|(k, v)| (k.as_str(), v))
        .collect();

    let inputs_root = ensure_fixtures().join("toolcalling/fixtures-stream-v2/inputs");
    assert!(inputs_root.is_dir(), "missing {}", inputs_root.display());

    // Discover fixture families (input subdirs) and load every case per family.
    let mut families: BTreeMap<String, BTreeMap<String, Case>> = BTreeMap::new();
    let mut files = Vec::new();
    collect_yaml(&inputs_root, &mut files);
    files.sort();
    for path in &files {
        let yaml = std::fs::read_to_string(path).unwrap();
        let fx: Fixture = match serde_yaml::from_str(&yaml) {
            Ok(f) => f,
            Err(e) => panic!("{}: YAML parse error: {e}", path.display()),
        };
        if !matches!(fx.mode.as_deref(), Some("stream" | "streamv2")) {
            continue;
        }
        // Family = parent dir name (the registry key).
        let family = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .expect("fixture parent dir")
            .to_string();
        families.entry(family).or_default().extend(fx.cases);
    }

    let mut ran_pairs = 0usize;
    let mut ran_families: Vec<String> = Vec::new();
    let mut skipped_families: Vec<String> = Vec::new();
    let mut errored_cases = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (family, cases) in &families {
        let Some(row) = enabled.get(family.as_str()) else {
            skipped_families.push(format!("{family} ({} cases, dynamo_v2=null)", cases.len()));
            continue;
        };
        let use_tokens = row.preferred_input == "tokens";
        ran_families.push(family.clone());

        let ids: Vec<String> = cases.keys().cloned().collect(); // BTreeMap => sorted
        for (i, j) in pairs_for(&ids) {
            let ca = &cases[&ids[i]];
            let cb = &cases[&ids[j]];
            let label = format!("{family} {}x{}", ids[i], ids[j]);
            for schedule in schedules() {
                let result = if use_tokens {
                    let sa: Vec<Vec<u32>> = ca
                        .chunks
                        .iter()
                        .map(|c| c.delta_token_ids.clone())
                        .collect();
                    let sb: Vec<Vec<u32>> = cb
                        .chunks
                        .iter()
                        .map(|c| c.delta_token_ids.clone())
                        .collect();
                    check_pair(
                        family,
                        &ca.tools,
                        &sa,
                        &cb.tools,
                        &sb,
                        schedule,
                        token_input,
                        &label,
                        &mut failures,
                    )
                } else {
                    let sa: Vec<String> = ca.chunks.iter().map(|c| c.delta_text.clone()).collect();
                    let sb: Vec<String> = cb.chunks.iter().map(|c| c.delta_text.clone()).collect();
                    check_pair(
                        family,
                        &ca.tools,
                        &sa,
                        &cb.tools,
                        &sb,
                        schedule,
                        text_input,
                        &label,
                        &mut failures,
                    )
                };
                match result {
                    Ok(()) => ran_pairs += 1,
                    // A parser error on a specific shape is logged + skipped, not
                    // silently passed — this lane is about isolation, not making
                    // every corpus shape parseable.
                    Err(e) => {
                        errored_cases += 1;
                        eprintln!(
                            "SKIP {label} schedule={}: parser error: {e}",
                            schedule.label()
                        );
                    }
                }
            }
        }
    }

    eprintln!(
        "v2 interleave isolation: {ran_pairs} pair-schedules over families [{}]",
        ran_families.join(", ")
    );
    for s in &skipped_families {
        eprintln!("SKIP family {s}");
    }
    eprintln!(
        "skipped {} families, {errored_cases} pair-schedules errored; elapsed {:.1}s",
        skipped_families.len(),
        started.elapsed().as_secs_f64()
    );

    assert!(ran_pairs > 0, "no enabled families exercised");
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        panic!("{} pair-schedules diverged", failures.len());
    }
}
