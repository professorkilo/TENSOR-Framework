# TENSOR Framework repository

Welcome to the public home of the **TENSOR Framework**. This repository holds the working source (“drafts”) and the immutable versioned releases of the Framework’s core definitions as well as optional extensions provided by the community.

## What is TENSOR?

* **TENSOR Framework** – A graph based security operations methodology that models every security fact as time‑series data. It aims to give analysts an extensible, technology‑neutral way to capture, correlate and respond to events.
* **TENSOR Standards Consortium** – A volunteer community that stewards the Framework, defines version policy and reviews proposed extensions. The group is currently organising and has not yet been incorporated.

## Repository layout

```
TENSOR-Framework/
├── drafts/                 # files that humans edit
│   ├── core/               # canonical schema + graph
│   ├── extensions/         # consortium-maintained add-ons
│   ├── vendors/<vendor>/<pack>/
│   └── orgs/<org>/<module>/
│       └── tensor.<...>.(schema|graph).json
├── releases/               # immutable machine-consumable snapshots
│   ├── manifest.json       # stable release catalog for clients
│   ├── checksums.json      # sha256 for release artifacts
│   └── core/
│       ├── graphs/
│       │   ├── latest/tensor.core.graph.json
│       │   └── v<VER>/tensor.core.graph.json
│       └── schemas/
│           ├── latest/tensor.core.schema.json
│           └── v<VER>/tensor.core.schema.json
└── standards/
    ├── compatibility-policy.md
    └── conformance/
        ├── manifest.json
        └── fixtures/(valid|invalid)/*.json
```

### Version string

`<VER>` follows `MAJOR.YYYYMMDD[letter]` (example: `1.20250910`). The same string appears inside each JSON as `"version"` and in the Git tag `core-v1.20250910` or `cloud-v1.20250712`.

### File‑name conventions

| Draft files                                | Purpose                                          |
| ------------------------------------------ | ------------------------------------------------ |
| `tensor.<namespace>.<package>.schema.json` | JSON Schema that extends Core.                   |
| `tensor.<namespace>.<package>.graph.json`  | Graph‑ontology file connecting the new concepts. |

| Release snapshots *(inside `releases/…/(schemas|graphs)/v<VER>/`)* | Purpose                              |
| ------------------------------------------------- | ------------------------------------ |
| `tensor.<namespace>.<package>.schema.json`        | Immutable schema for version `<VER>` |
| `tensor.<namespace>.<package>.graph.json`         | Immutable graph for version `<VER>`  |

* **`<namespace>`** = `core`, an official extension name (`cloud`), a vendor slug (`crowdstrike`), or an org slug (`acmebank`).
* **`<package>`**   = specific pack or module (`xdr`, `fraud`, etc.).
* File names never contain the version string; versioning is conveyed by the enclosing `v<VER>` folder.

### Editing basics

1. Make changes only inside `drafts/`.
2. Update `"version"` in both the schema and graph when you make a release breaking change (increment `MAJOR`) or a non‑breaking improvement (same `MAJOR`, new date).
3. When creating a release, snapshot drafts into `releases/<...>/(schemas|graphs)/v<VER>/`, update `releases/manifest.json`, refresh `releases/core/*/latest/*`, then regenerate checksums.
4. Run `python3 scripts/generate_release_checksums.py` and `python3 scripts/validate_release_contract.py` before opening a PR.

### Machine-readable release contract

* `releases/manifest.json` is the stable catalog for website/Graph Studio clients.
* `releases/core/graphs/latest/tensor.core.graph.json` and `releases/core/schemas/latest/tensor.core.schema.json` are pointer artifacts for default client loading.
* `releases/checksums.json` provides sha256 integrity metadata for all JSON release artifacts under `releases/`.

### Semantic governance and overlays

* Core (`releases/core/**`) remains vendor-neutral and defines canonical investigative semantics.
* Business logic belongs in overlays (`extensions/`, `vendors/`, `orgs/`) that extend Core without redefining Core meaning.
* Compatibility/deprecation requirements are defined in `standards/compatibility-policy.md`.

### Contributing extensions

* **Official extensions** live under `drafts/extensions/` and mirror Core rules.
* **Vendor packs** must sit one directory deeper under `drafts/vendors/<vendor>/<pack>/`.
* **Org overlays** follow the same pattern in `drafts/orgs/<org>/<module>/`.

Each layer supplies its own schema and graph file that extend, but do not modify, the Core definitions.

For detailed contribution guidelines, version policy and governance minutes see the forthcoming `CONTRIBUTING.md` and `GOVERNANCE.md` documents.
