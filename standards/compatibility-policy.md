# TENSOR Core Compatibility Policy

## Scope and governance
- `releases/core/**` is the normative Core release surface for machine consumers.
- Core investigative semantics are governed by the TENSOR standards process and must stay vendor-neutral.
- Vendor, product, and organization-specific logic must be implemented as overlays in extension layers (`extensions/`, `vendors/`, `orgs/`) and must not rewrite Core meaning.

## Layer model
- **Core layer**: canonical, stable investigative primitives and relationships.
- **Overlay layers**: additive constraints, fields, mappings, and workflow data for extensions/vendors/orgs.
- Overlay layers may reference Core nodes/edges and add metadata, but they cannot remove or redefine Core semantics.

## Versioning
- Version format: `<MAJOR>.<YYYYMMDD>[letter]`.
- Non-breaking Core changes publish a new date/revision within the current major.
- Breaking Core changes increment `MAJOR` and publish updated schema + graph artifacts.
- Release metadata is published in `releases/manifest.json`; `latest*` pointers identify the default client target.

## Compatibility expectations
- Clients that pin explicit versions (`graphVersion`, `schemaVersion`) must receive immutable artifacts.
- `latest/` files are convenience pointers and always mirror the versions declared by `releases/manifest.json`.
- Backward-compatible fields should be additive and optional whenever possible.

## Deprecation policy
- A deprecated Core element must be marked in release notes before removal in a later major.
- Deprecated elements remain available for at least one subsequent non-breaking release cycle.
- Removals/behavioral changes require a major version bump and explicit migration notes.

## Conformance
- Release contract conformance fixtures are in `standards/conformance/`.
- CI must validate manifest shape, version formatting, latest pointer integrity, and checksum integrity before merge.
