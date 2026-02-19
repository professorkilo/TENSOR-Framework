# TENSOR Core Final Pass Review (2026-02-12)

## Final Snapshot
- Version: `0.20260212e`
- Nodes: `219`
- Edges: `576`
- Entry nodes: `7` (`Q1..Q7`)
- Terminal nodes: `27`
- Cross-domain ratio: `0.644097`
- Publish gates: `PASS`
- Routing centralization: `0.015775` (threshold `<= 0.02`)
- Targeted resilience: `0.990385` (threshold `>= 0.85`)

## What Changed In This Final Pass
- Applied two centralization-headroom rewires:
  - `Q414-unknown` retargeted from `Q496` to `Q490`
  - `Q421-unknown` retargeted from `Q496` to `Q502`
- Consolidated remaining duplicate business-exposure variants:
  - Removed `Q514` and reassigned its incoming edges across `Q466`, `Q492`, `Q510`
  - Removed `Q515` and reassigned its incoming edges across `Q467`, `Q493`, `Q511`
- Canonicalized merged node phrasing:
  - `Q466`: `Identity: Business exposure`
  - `Q467`: `Host: Business exposure`
- Updated non-terminal fan-in exception list:
  - `Q467`, `Q492`, `Q493`, `Q494`, `Q506`, `Q507`, `Q508`, `Q509`, `Q581`

## Expected Reviewer Focus
- Verify technical-only language remains clear and objective.
- Verify no hidden “incident type” semantics are implied by merged impact nodes.
- Verify readability of key merged nodes in graph view:
  - `Q466` (`Identity: Business exposure`)
  - `Q467` (`Host: Business exposure`)
  - `Q507` (`Cloud: Business exposure`)
  - `Q508` (`Email: Business exposure`)
  - `Q509` (`Application: Business exposure`)
- Verify branch intent remains understandable for humans at each step (`yes/no/unknown`).

## Key Node/Edge Examples
- `Q5 (Email detect) ->yes-> Q154 ->yes-> Q161 ->yes-> Q299 ->yes-> Q372 ->yes-> Q408 ->yes-> Q470 ->yes-> Q508`
- `Q508 ->yes-> Q549`, `Q508 ->no-> Q546`, `Q508 ->unknown-> Q545`
- `Q467 ->yes-> Q546`, `Q467 ->no-> Q550`, `Q467 ->unknown-> Q549`
- Centralization headroom rewires in this pass:
  - `Q414-unknown -> Q490`
  - `Q421-unknown -> Q502`

## Files Updated
- `drafts/core/source/nodes.jsonl`
- `drafts/core/source/edges.jsonl`
- `drafts/core/source/entry_nodes.json`
- `drafts/core/tensor.core.graph.json`
