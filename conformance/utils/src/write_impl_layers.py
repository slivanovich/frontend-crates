#!/usr/bin/env python3
"""Write a peer impl's anchor + changed-only overlay trees from two captures.

Corpus layout: the LOWEST published version of an impl is the full anchor; every higher
version stores only the cases whose output DIFFERS from the anchor. Re-capturing both
versions and diffing is the only way to keep that invariant true — updating just the
cases an overlay already happens to contain silently keeps stale entries and drops cases
that newly diverged.

Usage: write_layers.py <anchor.json> <anchor-dir> <over.json> <over-dir> <impl> [--apply]
"""
import json
import pathlib
import sys

import yaml

anchor_cap, anchor_dir, over_cap, over_dir, impl = sys.argv[1:6]
APPLY = "--apply" in sys.argv


def load(path):
    cap = json.load(open(path))
    out = {}
    for fx, body in cap["fixtures"].items():
        p = pathlib.Path(fx)
        for cid, res in (body.get("cases") or {}).items():
            out[(p.parent.name, p.name, cid)] = res
    return cap["version"], out


def case_body(res):
    """Capture result -> the fields an overlay stores for one case."""
    if isinstance(res, dict) and "error" in res:
        return {"unavailable": f"{impl} parser not captured: {res['error']}"}
    return {"chunks": [{"expected": ch.get("deltas") or []} for ch in res]}


av, anchor = load(anchor_cap)
ov, over = load(over_cap)
print(f"{impl}: anchor={av} ({len(anchor)} cases)  overlay={ov} ({len(over)} cases)")

written = {"anchor": 0, "overlay": 0}
for label, tree, cap, other in (("anchor", anchor_dir, anchor, None),
                                ("overlay", over_dir, over, anchor)):
    root = pathlib.Path(tree)
    for f in sorted(root.glob("*/TOOLCALLING.streamv2*.yaml")):
        doc = yaml.safe_load(f.read_text()) or {}
        keep = {}
        for cid in list((doc.get("cases") or {}).keys()):
            key = (f.parent.name, f.name, cid)
            res = cap.get(key)
            if res is None:
                keep[cid] = doc["cases"][cid]        # impl not applicable here
                continue
            body = case_body(res)
            if other is not None:
                base = other.get(key)
                if base is not None and case_body(base) == body:
                    continue                          # identical to anchor -> omit
            keep[cid] = body
            written[label] += 1
        doc["cases"] = keep
        # `captured_with` names the version this tree speaks for.
        doc.setdefault("captured_with", {})[impl] = av if label == "anchor" else ov
        out = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=4096)
        assert yaml.safe_load(out) == doc, f"{f}: round-trip mismatch"
        if APPLY:
            f.write_text(out)
print(f"  cases written: anchor={written['anchor']}  overlay(changed-only)={written['overlay']}")
