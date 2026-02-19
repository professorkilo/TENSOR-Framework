# Core vs Overlay Contract (TENSOR)

## Purpose

TENSOR Core exists to guide investigations with a vendor-neutral, technical, evidence-first graph.
Overlays exist to implement environment-specific execution and business/product process on top of Core
without redefining Core meaning.

This contract is the line between:
- **Core**: shared technical language + guard rails + deterministic routing primitives
- **Overlays**: tool bindings, automation, and organizational process

## Core (Allowed / Required)

Core MUST:
- Express **technical, evidence-checkable questions** about observable facts.
- Route only via `yes` / `no` / `unknown` decisions.
- Support cross-domain pivots **only when justified by technical facts**.
- Remain **tool- and vendor-neutral**.
- Remain **process-neutral** (no case/ticket lifecycle, no assignments, no response procedures).

Core node text MUST:
- Be an objective question ending with `?`.
- Describe *what can be established with evidence*, not what to do operationally.

Core node labels MUST:
- Be short and graph-readable (optimized for scanning in a visual DAG).
- Start with the node's domain in the form `<Domain>:` (for example: `Email:`).
- Describe a technical signal (not an incident type and not a business/process state).
- Avoid generic numeric templates (for example: `Email detect 12`).

Core MAY include:
- Technical assertions such as authenticity, provenance, correlation, scope, attribution of activity,
  and measurable impact (still stated as questions).
- Generic technical qualifiers like "malicious" / "benign" / "verified" only when they are grounded
  in objective indicators (hash matches, signatures, sandbox behavior, protocol traces, etc.).

## Core (Forbidden)

Core MUST NOT include:
- **Incident-type labels** (for example: phishing, BEC, ransomware) as investigation outcomes or branches.
- **"What is it?"** / case typing semantics (for example: "incident type", "case type", "categorize this incident").
- Business workflow state (for example: ticket status, SLA, escalation, severity workflows).
- Tool/product execution steps (for example: "run query in X", "click dashboard", "open console").
- Vendor-specific playbooks or environment-specific procedures.

Rationale: Core is the shared semantic backbone. If Core encodes incident typing or workflow, it stops being
portable and starts biasing investigations before evidence is established.

## Overlays (Allowed / Expected)

Overlays SHOULD:
- Bind Core questions to concrete data sources, queries, detections, enrichments, and automation.
- Add business and product process (case management, response workflows, notifications, SLAs).
- Add organizational taxonomies (including incident-type naming) **only after** technical evidence exists.
- Add additional entry signals and shortcuts without altering Core decision semantics.

Overlays MUST:
- Use namespaced extension keys (`<namespace>:<field>`) to avoid leaking business/process metadata into Core meaning.
- Avoid redefining Core node/edge semantics; overlays should extend, not replace, Core semantics.

## Examples

### Core-style node (good)
- "Does the message contain an attachment?"

### Overlay-style execution (good, but overlay-only)
- "In Microsoft 365 Defender, query `EmailAttachmentInfo` for attachments for the message id and compute SHA256."

### Core-style pivot (good)
- Attachment present (`yes`) -> "Is the attachment malicious by objective indicators?"
- Attachment malicious (`yes`) -> File domain: "Is there a corresponding file artifact on any endpoint?"

### Forbidden in Core (move to overlays)
- "Is this phishing?"
- "Classify into an actionable investigation category."
- "Escalate to IR and open a Sev-1 ticket."
