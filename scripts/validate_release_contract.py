#!/usr/bin/env python3
"""Validate TENSOR release contract integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASES_DIR = REPO_ROOT / "releases"
MANIFEST_PATH = RELEASES_DIR / "manifest.json"
CHECKSUMS_PATH = RELEASES_DIR / "checksums.json"

TOP_LEVEL_KEYS = [
    "channel",
    "generatedAt",
    "latestGraphVersion",
    "latestSchemaVersion",
    "releases",
]

RELEASE_KEYS = [
    "id",
    "displayName",
    "releasedAt",
    "graphVersion",
    "schemaVersion",
    "graphPath",
    "schemaPath",
    "notes",
]

VERSION_RE = re.compile(r"^\d+\.\d{8}[a-z]?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NODE_ID_RE = re.compile(r"^Q[1-9]\d*$")
EDGE_ID_RE = re.compile(r"^(Q[1-9]\d*)-(yes|no|unknown)-(Q[1-9]\d*)$")
DECISIONS = {"yes", "no", "unknown"}
ENFORCEMENT_VERSION = (0, 20260206, "")
ARCHETYPE_ENFORCEMENT_VERSION = (0, 20260206, "c")
MATH_ASSURANCE_ENFORCEMENT_VERSION = (0, 20260206, "d")
PUBLISH_GATES_ENFORCEMENT_VERSION = (0, 20260206, "e")
REQUIRED_ARCHETYPES = {
    "detect",
    "validate",
    "classify",
    "scope",
    "correlate",
    "attribute",
    "impact",
    "terminal",
}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing required file: {path.relative_to(REPO_ROOT)}") from exc
    except json.JSONDecodeError as exc:
        rel = path.relative_to(REPO_ROOT)
        raise RuntimeError(f"Invalid JSON in {rel}: {exc.msg} at line {exc.lineno}") from exc


def is_iso_datetime(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if not re.search(r"(Z|[+-]\d{2}:\d{2})$", value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def parse_version(value: str) -> tuple[int, int, str] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"^(\d+)\.(\d{8})([a-z]?)$", value)
    if not match:
        return None
    major = int(match.group(1))
    yyyymmdd = int(match.group(2))
    rev = match.group(3) or ""
    return (major, yyyymmdd, rev)


def version_at_least(value: str, threshold: tuple[int, int, str]) -> bool:
    parsed = parse_version(value)
    if parsed is None:
        return False
    return parsed >= threshold


def node_sort_key(node_id: str) -> tuple[int, str]:
    try:
        return (int(node_id[1:]), node_id)
    except (ValueError, IndexError):
        return (sys.maxsize, node_id)


def unwrap_graph_item(item: Any) -> tuple[dict[str, Any] | None, str]:
    """Return payload and shape marker ('flat' or 'wrapped')."""
    if not isinstance(item, dict):
        return (None, "invalid")
    if "data" in item:
        payload = item.get("data")
        if isinstance(payload, dict):
            return (payload, "wrapped")
        return (None, "invalid")
    return (item, "flat")


def compute_release_checksums() -> dict[str, str]:
    checksums: dict[str, str] = {}
    for artifact in sorted(RELEASES_DIR.rglob("*.json")):
        if artifact.resolve() == CHECKSUMS_PATH.resolve():
            continue
        rel = artifact.relative_to(REPO_ROOT).as_posix()
        checksums[rel] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return checksums


def validate_manifest(manifest: Any) -> list[str]:
    errors: list[str] = []

    if not isinstance(manifest, dict):
        return ["Manifest root must be a JSON object."]

    top_keys = set(manifest.keys())
    if top_keys != set(TOP_LEVEL_KEYS):
        missing = sorted(set(TOP_LEVEL_KEYS) - top_keys)
        extra = sorted(top_keys - set(TOP_LEVEL_KEYS))
        if missing:
            errors.append(f"Manifest missing top-level keys: {', '.join(missing)}")
        if extra:
            errors.append(f"Manifest has unsupported top-level keys: {', '.join(extra)}")

    channel = manifest.get("channel")
    if not isinstance(channel, str) or not channel.strip():
        errors.append("Manifest 'channel' must be a non-empty string.")

    generated_at = manifest.get("generatedAt")
    if not is_iso_datetime(generated_at):
        errors.append("Manifest 'generatedAt' must be an ISO 8601 datetime with timezone.")

    latest_graph_version = manifest.get("latestGraphVersion")
    latest_schema_version = manifest.get("latestSchemaVersion")

    if not isinstance(latest_graph_version, str) or not VERSION_RE.fullmatch(latest_graph_version):
        errors.append("Manifest 'latestGraphVersion' is malformed.")
    if not isinstance(latest_schema_version, str) or not VERSION_RE.fullmatch(latest_schema_version):
        errors.append("Manifest 'latestSchemaVersion' is malformed.")

    releases = manifest.get("releases")
    if not isinstance(releases, list) or not releases:
        errors.append("Manifest 'releases' must be a non-empty array.")
        return errors

    release_ids: set[str] = set()
    graph_versions: set[str] = set()
    schema_versions: set[str] = set()

    for idx, release in enumerate(releases):
        context = f"releases[{idx}]"

        if not isinstance(release, dict):
            errors.append(f"{context} must be an object.")
            continue

        keys = set(release.keys())
        if keys != set(RELEASE_KEYS):
            missing = sorted(set(RELEASE_KEYS) - keys)
            extra = sorted(keys - set(RELEASE_KEYS))
            if missing:
                errors.append(f"{context} missing keys: {', '.join(missing)}")
            if extra:
                errors.append(f"{context} has unsupported keys: {', '.join(extra)}")

        release_id = release.get("id")
        if not isinstance(release_id, str) or not release_id.strip():
            errors.append(f"{context}.id must be a non-empty string.")
        elif release_id in release_ids:
            errors.append(f"Duplicate release id: {release_id}")
        else:
            release_ids.add(release_id)

        display_name = release.get("displayName")
        if not isinstance(display_name, str) or not display_name.strip():
            errors.append(f"{context}.displayName must be a non-empty string.")

        released_at = release.get("releasedAt")
        if not isinstance(released_at, str) or not DATE_RE.fullmatch(released_at):
            errors.append(f"{context}.releasedAt must use YYYY-MM-DD.")
        else:
            try:
                datetime.strptime(released_at, "%Y-%m-%d")
            except ValueError:
                errors.append(f"{context}.releasedAt is not a valid date: {released_at}")

        graph_version = release.get("graphVersion")
        schema_version = release.get("schemaVersion")

        if not isinstance(graph_version, str) or not VERSION_RE.fullmatch(graph_version):
            errors.append(f"{context}.graphVersion is malformed.")
        else:
            graph_versions.add(graph_version)

        if not isinstance(schema_version, str) or not VERSION_RE.fullmatch(schema_version):
            errors.append(f"{context}.schemaVersion is malformed.")
        else:
            schema_versions.add(schema_version)

        graph_path = release.get("graphPath")
        schema_path = release.get("schemaPath")

        if isinstance(graph_version, str) and VERSION_RE.fullmatch(graph_version):
            expected_graph_path = f"releases/core/graphs/v{graph_version}/tensor.core.graph.json"
            if graph_path != expected_graph_path:
                errors.append(
                    f"{context}.graphPath must be '{expected_graph_path}' (found '{graph_path}')."
                )

        if isinstance(schema_version, str) and VERSION_RE.fullmatch(schema_version):
            expected_schema_path = f"releases/core/schemas/v{schema_version}/tensor.core.schema.json"
            if schema_path != expected_schema_path:
                errors.append(
                    f"{context}.schemaPath must be '{expected_schema_path}' (found '{schema_path}')."
                )

        if not isinstance(graph_path, str) or graph_path.startswith("/"):
            errors.append(f"{context}.graphPath must be a repo-relative path.")
        else:
            graph_file = REPO_ROOT / graph_path
            if not graph_file.is_file():
                errors.append(f"{context}.graphPath references missing file: {graph_path}")

        if not isinstance(schema_path, str) or schema_path.startswith("/"):
            errors.append(f"{context}.schemaPath must be a repo-relative path.")
        else:
            schema_file = REPO_ROOT / schema_path
            if not schema_file.is_file():
                errors.append(f"{context}.schemaPath references missing file: {schema_path}")

        notes = release.get("notes")
        if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
            errors.append(f"{context}.notes must be an array of strings.")

    if isinstance(latest_graph_version, str) and VERSION_RE.fullmatch(latest_graph_version):
        if latest_graph_version not in graph_versions:
            errors.append(
                "Manifest latestGraphVersion is not present in releases[].graphVersion."
            )

    if isinstance(latest_schema_version, str) and VERSION_RE.fullmatch(latest_schema_version):
        if latest_schema_version not in schema_versions:
            errors.append(
                "Manifest latestSchemaVersion is not present in releases[].schemaVersion."
            )

    if (
        isinstance(latest_graph_version, str)
        and VERSION_RE.fullmatch(latest_graph_version)
        and isinstance(latest_schema_version, str)
        and VERSION_RE.fullmatch(latest_schema_version)
    ):
        if not any(
            isinstance(r, dict)
            and r.get("graphVersion") == latest_graph_version
            and r.get("schemaVersion") == latest_schema_version
            for r in releases
        ):
            errors.append(
                "Manifest must include at least one release matching latestGraphVersion/latestSchemaVersion."
            )

        expected_graph = RELEASES_DIR / "core" / "graphs" / f"v{latest_graph_version}" / "tensor.core.graph.json"
        expected_schema = RELEASES_DIR / "core" / "schemas" / f"v{latest_schema_version}" / "tensor.core.schema.json"
        latest_graph = RELEASES_DIR / "core" / "graphs" / "latest" / "tensor.core.graph.json"
        latest_schema = RELEASES_DIR / "core" / "schemas" / "latest" / "tensor.core.schema.json"

        if not expected_graph.is_file():
            errors.append(
                f"Expected latest graph file is missing: {expected_graph.relative_to(REPO_ROOT)}"
            )
        if not expected_schema.is_file():
            errors.append(
                f"Expected latest schema file is missing: {expected_schema.relative_to(REPO_ROOT)}"
            )
        if not latest_graph.is_file():
            errors.append("Latest graph pointer file is missing: releases/core/graphs/latest/tensor.core.graph.json")
        if not latest_schema.is_file():
            errors.append("Latest schema pointer file is missing: releases/core/schemas/latest/tensor.core.schema.json")

        if expected_graph.is_file() and latest_graph.is_file():
            if latest_graph.read_bytes() != expected_graph.read_bytes():
                errors.append(
                    "Latest graph pointer does not match releases/core/graphs/"
                    f"v{latest_graph_version}/tensor.core.graph.json"
                )

        if expected_schema.is_file() and latest_schema.is_file():
            if latest_schema.read_bytes() != expected_schema.read_bytes():
                errors.append(
                    "Latest schema pointer does not match releases/core/schemas/"
                    f"v{latest_schema_version}/tensor.core.schema.json"
                )

    return errors


def schema_supports_entry_nodes(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return False
    return "entryNodeIds" in properties


def schema_enforces_source_target_node_refs(schema: Any) -> bool:
    found_source = False
    found_target = False

    def walk(value: Any) -> None:
        nonlocal found_source, found_target
        if isinstance(value, dict):
            for key, nested in value.items():
                if (
                    key == "source"
                    and isinstance(nested, dict)
                    and nested.get("$ref") == "#/$defs/nodeId"
                ):
                    found_source = True
                if (
                    key == "target"
                    and isinstance(nested, dict)
                    and nested.get("$ref") == "#/$defs/nodeId"
                ):
                    found_target = True
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(schema)
    return found_source and found_target


def schema_supports_flat_and_wrapped_items(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    text = json.dumps(schema, sort_keys=True)
    return all(
        token in text
        for token in (
            "#/$defs/nodeFlat",
            "#/$defs/nodeWrapped",
            "#/$defs/edgeFlat",
            "#/$defs/edgeWrapped",
        )
    )


def schema_enforces_namespaced_extensions(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return False
    ext_obj = defs.get("extensionsObject")
    if not isinstance(ext_obj, dict):
        return False
    patterns = ext_obj.get("patternProperties")
    if not isinstance(patterns, dict):
        return False
    return any(":" in pattern for pattern in patterns)


def schema_requires_node_archetype(schema: Any, required_archetypes: set[str]) -> bool:
    if not isinstance(schema, dict):
        return False
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return False
    node_flat = defs.get("nodeFlat")
    if not isinstance(node_flat, dict):
        return False
    required = node_flat.get("required")
    if not isinstance(required, list) or "archetype" not in required:
        return False
    properties = node_flat.get("properties")
    if not isinstance(properties, dict):
        return False
    archetype = properties.get("archetype")
    if not isinstance(archetype, dict):
        return False
    enum = archetype.get("enum")
    if not isinstance(enum, list):
        return False
    return set(enum) == required_archetypes


def validate_graph_semantics(
    graph: Any,
    schema: Any,
    release: dict[str, Any],
    context: str,
) -> list[str]:
    errors: list[str] = []
    graph_version = release.get("graphVersion")
    schema_version = release.get("schemaVersion")

    if not isinstance(graph, dict):
        return [f"{context}.graphPath must contain a JSON object root."]

    if graph.get("namespace") != "tensor":
        errors.append(f"{context}.graphPath must use namespace='tensor'.")
    if graph.get("product") != "core":
        errors.append(f"{context}.graphPath must use product='core'.")

    if graph.get("version") != graph_version:
        errors.append(
            f"{context}.graphPath version ({graph.get('version')!r}) does not match graphVersion ({graph_version!r})."
        )
    if graph.get("schemaVersion") != schema_version:
        errors.append(
            f"{context}.graphPath schemaVersion ({graph.get('schemaVersion')!r}) does not match schemaVersion ({schema_version!r})."
        )

    if not is_iso_datetime(graph.get("generatedAt")):
        errors.append(f"{context}.graphPath generatedAt must be an ISO 8601 datetime with timezone.")

    nodes_raw = graph.get("nodes")
    edges_raw = graph.get("edges")

    if not isinstance(nodes_raw, list) or not nodes_raw:
        errors.append(f"{context}.graphPath nodes must be a non-empty array.")
        nodes_raw = []

    if not isinstance(edges_raw, list):
        errors.append(f"{context}.graphPath edges must be an array.")
        edges_raw = []

    node_ids: set[str] = set()
    node_categories: dict[str, str] = {}
    edge_ids: set[str] = set()
    parsed_edges: list[dict[str, str]] = []
    seen_branch_tuples: set[tuple[str, str, str]] = set()
    seen_source_decisions: set[tuple[str, str]] = set()
    outgoing_by_source: dict[str, set[str]] = defaultdict(set)

    requires_archetype = (
        isinstance(graph_version, str)
        and version_at_least(graph_version, ARCHETYPE_ENFORCEMENT_VERSION)
    )
    required_archetypes = REQUIRED_ARCHETYPES if requires_archetype else set()

    wrapped_nodes = 0
    for idx, raw_node in enumerate(nodes_raw):
        payload, shape = unwrap_graph_item(raw_node)
        if payload is None:
            errors.append(f"{context}.graphPath nodes[{idx}] must be an object (flat or wrapped under data).")
            continue
        if shape == "wrapped":
            wrapped_nodes += 1

        node_id = payload.get("id")
        if not isinstance(node_id, str):
            errors.append(f"{context}.graphPath nodes[{idx}] id must be a string.")
            continue
        if not NODE_ID_RE.fullmatch(node_id):
            errors.append(f"{context}.graphPath nodes[{idx}] id is malformed: {node_id!r}")
            continue
        if node_id in node_ids:
            errors.append(f"{context}.graphPath has duplicate node id: {node_id}")
            continue
        node_ids.add(node_id)

        category = payload.get("category")
        if isinstance(category, str):
            node_categories[node_id] = category
        elif category is not None:
            errors.append(f"{context}.graphPath nodes[{idx}] category must be a string when provided.")

        if requires_archetype:
            archetype = payload.get("archetype")
            if not isinstance(archetype, str) or archetype not in required_archetypes:
                allowed = ", ".join(sorted(required_archetypes))
                errors.append(
                    f"{context}.graphPath nodes[{idx}] archetype must be one of [{allowed}] for this release."
                )

    wrapped_edges = 0
    for idx, raw_edge in enumerate(edges_raw):
        payload, shape = unwrap_graph_item(raw_edge)
        if payload is None:
            errors.append(f"{context}.graphPath edges[{idx}] must be an object (flat or wrapped under data).")
            continue
        if shape == "wrapped":
            wrapped_edges += 1

        edge_id = payload.get("id")
        source = payload.get("source")
        target = payload.get("target")
        decision = payload.get("decision")

        if not isinstance(edge_id, str):
            errors.append(f"{context}.graphPath edges[{idx}] id must be a string.")
            continue
        if edge_id in edge_ids:
            errors.append(f"{context}.graphPath has duplicate edge id: {edge_id}")
            continue
        edge_ids.add(edge_id)

        if not EDGE_ID_RE.fullmatch(edge_id):
            errors.append(f"{context}.graphPath edges[{idx}] id is malformed: {edge_id!r}")

        if not isinstance(source, str):
            errors.append(f"{context}.graphPath edges[{idx}] source must be a string.")
            continue
        if not isinstance(target, str):
            errors.append(f"{context}.graphPath edges[{idx}] target must be a string.")
            continue
        if not isinstance(decision, str):
            errors.append(f"{context}.graphPath edges[{idx}] decision must be a string.")
            continue

        if not NODE_ID_RE.fullmatch(source):
            errors.append(f"{context}.graphPath edges[{idx}] source id is malformed: {source!r}")
        if not NODE_ID_RE.fullmatch(target):
            errors.append(f"{context}.graphPath edges[{idx}] target id is malformed: {target!r}")
        if decision not in DECISIONS:
            errors.append(f"{context}.graphPath edges[{idx}] decision must be one of yes/no/unknown.")
        if requires_archetype and NODE_ID_RE.fullmatch(source) and NODE_ID_RE.fullmatch(target):
            if int(target[1:]) <= int(source[1:]):
                errors.append(
                    f"{context}.graphPath edges[{idx}] must point to a higher node id to preserve DAG ordering."
                )

        match = EDGE_ID_RE.fullmatch(edge_id)
        if match and (source != match.group(1) or decision != match.group(2) or target != match.group(3)):
            errors.append(
                f"{context}.graphPath edges[{idx}] id tuple must match source/decision/target for edge {edge_id!r}."
            )

        branch_tuple = (source, decision, target)
        if branch_tuple in seen_branch_tuples:
            errors.append(
                f"{context}.graphPath has duplicate branch tuple source={source}, decision={decision}, target={target}."
            )
        seen_branch_tuples.add(branch_tuple)
        source_decision = (source, decision)
        if requires_archetype:
            if source_decision in seen_source_decisions:
                errors.append(
                    f"{context}.graphPath has non-deterministic branching for source={source}, decision={decision}."
                )
            seen_source_decisions.add(source_decision)
        outgoing_by_source[source].add(decision)
        parsed_edges.append({"id": edge_id, "source": source, "target": target, "decision": decision})

    for edge in parsed_edges:
        if edge["source"] not in node_ids:
            errors.append(
                f"{context}.graphPath edge {edge['id']!r} references missing source node: {edge['source']}"
            )
        if edge["target"] not in node_ids:
            errors.append(
                f"{context}.graphPath edge {edge['id']!r} references missing target node: {edge['target']}"
            )

    requires_new_semantics = (
        isinstance(graph_version, str) and version_at_least(graph_version, ENFORCEMENT_VERSION)
    )

    entry_nodes = graph.get("entryNodeIds")
    if requires_new_semantics and not schema_supports_entry_nodes(schema):
        errors.append(f"{context}.schemaPath must define properties.entryNodeIds.")
    if requires_new_semantics and not schema_enforces_source_target_node_refs(schema):
        errors.append(
            f"{context}.schemaPath must constrain edge source/target to nodeId references."
        )
    if requires_new_semantics and not schema_supports_flat_and_wrapped_items(schema):
        errors.append(
            f"{context}.schemaPath must support both flat and wrapped node/edge formats during migration."
        )
    if requires_new_semantics and not schema_enforces_namespaced_extensions(schema):
        errors.append(
            f"{context}.schemaPath must enforce namespaced extension keys (<namespace>:<field>)."
        )
    if requires_archetype and not schema_requires_node_archetype(schema, required_archetypes):
        errors.append(
            f"{context}.schemaPath must require node archetype with the approved archetype enum."
        )
    if requires_new_semantics and wrapped_nodes > 0:
        errors.append(
            f"{context}.graphPath must use flat node objects for canonical Core releases (wrapped nodes found: {wrapped_nodes})."
        )
    if requires_new_semantics and wrapped_edges > 0:
        errors.append(
            f"{context}.graphPath must use flat edge objects for canonical Core releases (wrapped edges found: {wrapped_edges})."
        )

    if requires_new_semantics and "entryNodeIds" not in graph:
        errors.append(f"{context}.graphPath must define entryNodeIds for this release.")
    elif entry_nodes is not None:
        if not isinstance(entry_nodes, list) or not entry_nodes:
            errors.append(f"{context}.graphPath entryNodeIds must be a non-empty array.")
        else:
            seen_entries: set[str] = set()
            valid_entries: list[str] = []
            for idx, entry_node in enumerate(entry_nodes):
                if not isinstance(entry_node, str):
                    errors.append(
                        f"{context}.graphPath entryNodeIds[{idx}] must be a string node id."
                    )
                    continue
                if not NODE_ID_RE.fullmatch(entry_node):
                    errors.append(
                        f"{context}.graphPath entryNodeIds[{idx}] is malformed: {entry_node!r}"
                    )
                    continue
                if entry_node in seen_entries:
                    errors.append(
                        f"{context}.graphPath entryNodeIds contains duplicate id: {entry_node}"
                    )
                    continue
                seen_entries.add(entry_node)
                if entry_node not in node_ids:
                    errors.append(
                        f"{context}.graphPath entryNodeIds references missing node: {entry_node}"
                    )
                    continue
                valid_entries.append(entry_node)

            if valid_entries and node_ids:
                adjacency: dict[str, list[str]] = defaultdict(list)
                for edge in parsed_edges:
                    if edge["source"] in node_ids and edge["target"] in node_ids:
                        adjacency[edge["source"]].append(edge["target"])

                reachable: set[str] = set()
                queue: deque[str] = deque()
                for entry in valid_entries:
                    if entry not in reachable:
                        reachable.add(entry)
                        queue.append(entry)

                while queue:
                    current = queue.popleft()
                    for nxt in adjacency.get(current, []):
                        if nxt not in reachable:
                            reachable.add(nxt)
                            queue.append(nxt)

                missing = sorted(node_ids - reachable, key=node_sort_key)
                if missing:
                    preview = ", ".join(missing[:10])
                    suffix = "..." if len(missing) > 10 else ""
                    errors.append(
                        f"{context}.graphPath has nodes unreachable from entryNodeIds: {preview}{suffix}"
                    )

    if requires_archetype:
        for node_id in node_ids:
            decisions = outgoing_by_source.get(node_id, set())
            if decisions and decisions != DECISIONS:
                errors.append(
                    f"{context}.graphPath node {node_id} must expose yes/no/unknown when non-terminal."
                )

    return errors


def validate_release_graph_semantics(manifest: Any) -> list[str]:
    errors: list[str] = []
    releases = manifest.get("releases") if isinstance(manifest, dict) else None
    if not isinstance(releases, list):
        return errors

    for idx, release in enumerate(releases):
        context = f"releases[{idx}]"
        if not isinstance(release, dict):
            continue
        graph_path = release.get("graphPath")
        schema_path = release.get("schemaPath")
        if not isinstance(graph_path, str) or not isinstance(schema_path, str):
            continue

        graph_file = REPO_ROOT / graph_path
        schema_file = REPO_ROOT / schema_path
        if not graph_file.is_file():
            continue

        graph_version = release.get("graphVersion")
        requires_new_semantics = (
            isinstance(graph_version, str)
            and version_at_least(graph_version, ENFORCEMENT_VERSION)
        )

        try:
            graph = load_json(graph_file)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue

        schema: Any = {}
        if requires_new_semantics:
            if not schema_file.is_file():
                errors.append(f"{context}.schemaPath references missing file: {schema_path}")
                continue
            try:
                schema = load_json(schema_file)
            except RuntimeError as exc:
                errors.append(str(exc))
                continue

        errors.extend(validate_graph_semantics(graph, schema, release, context))

    return errors


def validate_release_report_artifacts(manifest: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return errors

    latest_graph_version = manifest.get("latestGraphVersion")
    releases = manifest.get("releases")
    if not isinstance(releases, list):
        return errors

    required_report_files = [
        "graph-quality.json",
        "graph-quality.md",
        "coverage-matrix.json",
        "coverage-matrix.md",
        "math-assurance.json",
        "math-assurance.md",
    ]

    for idx, release in enumerate(releases):
        context = f"releases[{idx}]"
        if not isinstance(release, dict):
            continue
        graph_version = release.get("graphVersion")
        if not isinstance(graph_version, str):
            continue
        if not version_at_least(graph_version, MATH_ASSURANCE_ENFORCEMENT_VERSION):
            continue

        version_dir = REPO_ROOT / f"releases/core/reports/v{graph_version}"
        for filename in required_report_files:
            report_path = version_dir / filename
            if not report_path.is_file():
                errors.append(
                    f"{context} missing required report artifact: {report_path.relative_to(REPO_ROOT)}"
                )

        math_json_path = version_dir / "math-assurance.json"
        if math_json_path.is_file():
            try:
                payload = load_json(math_json_path)
            except RuntimeError as exc:
                errors.append(str(exc))
            else:
                if not isinstance(payload, dict):
                    errors.append(
                        f"{context} math assurance payload must be a JSON object: {math_json_path.relative_to(REPO_ROOT)}"
                    )
                else:
                    if payload.get("version") != graph_version:
                        errors.append(
                            f"{context} math assurance version mismatch in {math_json_path.relative_to(REPO_ROOT)}"
                        )
                    if "summary" not in payload or "theories" not in payload:
                        errors.append(
                            f"{context} math assurance payload missing summary/theories in {math_json_path.relative_to(REPO_ROOT)}"
                        )
                    if version_at_least(graph_version, PUBLISH_GATES_ENFORCEMENT_VERSION):
                        monitoring = payload.get("monitoring")
                        if not isinstance(monitoring, dict):
                            errors.append(
                                f"{context} math assurance payload missing monitoring object in {math_json_path.relative_to(REPO_ROOT)}"
                            )
                        else:
                            publish_gates = monitoring.get("publishGates")
                            if not isinstance(publish_gates, dict):
                                errors.append(
                                    f"{context} math assurance payload missing monitoring.publishGates in {math_json_path.relative_to(REPO_ROOT)}"
                                )
                            else:
                                all_passed = publish_gates.get("allPassed")
                                gates = publish_gates.get("gates")
                                failed_gate_ids = publish_gates.get("failedGateIds")
                                if not isinstance(all_passed, bool):
                                    errors.append(
                                        f"{context} monitoring.publishGates.allPassed must be boolean in {math_json_path.relative_to(REPO_ROOT)}"
                                    )
                                if not isinstance(gates, list) or not gates:
                                    errors.append(
                                        f"{context} monitoring.publishGates.gates must be a non-empty array in {math_json_path.relative_to(REPO_ROOT)}"
                                    )
                                if not isinstance(failed_gate_ids, list):
                                    errors.append(
                                        f"{context} monitoring.publishGates.failedGateIds must be an array in {math_json_path.relative_to(REPO_ROOT)}"
                                    )
                                if all_passed is False:
                                    rendered = ", ".join(str(item) for item in failed_gate_ids) if isinstance(failed_gate_ids, list) else "unknown"
                                    errors.append(
                                        f"{context} publish gates failed in {math_json_path.relative_to(REPO_ROOT)}: {rendered}"
                                    )

    if (
        isinstance(latest_graph_version, str)
        and VERSION_RE.fullmatch(latest_graph_version)
        and version_at_least(latest_graph_version, MATH_ASSURANCE_ENFORCEMENT_VERSION)
    ):
        expected_latest_math_json = REPO_ROOT / f"releases/core/reports/v{latest_graph_version}/math-assurance.json"
        expected_latest_math_md = REPO_ROOT / f"releases/core/reports/v{latest_graph_version}/math-assurance.md"
        latest_math_json = REPO_ROOT / "releases/core/reports/latest/math-assurance.json"
        latest_math_md = REPO_ROOT / "releases/core/reports/latest/math-assurance.md"
        history_json = REPO_ROOT / "releases/core/reports/history/math-assurance-history.json"
        history_md = REPO_ROOT / "releases/core/reports/history/math-assurance-history.md"

        if not latest_math_json.is_file():
            errors.append("Missing latest math assurance pointer: releases/core/reports/latest/math-assurance.json")
        if not latest_math_md.is_file():
            errors.append("Missing latest math assurance pointer: releases/core/reports/latest/math-assurance.md")
        if not history_json.is_file():
            errors.append("Missing math assurance history: releases/core/reports/history/math-assurance-history.json")
        if not history_md.is_file():
            errors.append("Missing math assurance history: releases/core/reports/history/math-assurance-history.md")

        if expected_latest_math_json.is_file() and latest_math_json.is_file():
            if expected_latest_math_json.read_bytes() != latest_math_json.read_bytes():
                errors.append(
                    "Latest math assurance JSON pointer does not match "
                    f"releases/core/reports/v{latest_graph_version}/math-assurance.json"
                )
        if expected_latest_math_md.is_file() and latest_math_md.is_file():
            if expected_latest_math_md.read_bytes() != latest_math_md.read_bytes():
                errors.append(
                    "Latest math assurance Markdown pointer does not match "
                    f"releases/core/reports/v{latest_graph_version}/math-assurance.md"
                )

        if history_json.is_file():
            try:
                history_payload = load_json(history_json)
            except RuntimeError as exc:
                errors.append(str(exc))
            else:
                if not isinstance(history_payload, dict):
                    errors.append("Math assurance history payload must be an object.")
                else:
                    series = history_payload.get("series")
                    if not isinstance(series, list):
                        errors.append("Math assurance history payload must define series as an array.")
                    else:
                        latest_series_entry = None
                        for item in series:
                            if isinstance(item, dict) and item.get("version") == latest_graph_version:
                                latest_series_entry = item
                                break
                        if latest_series_entry is None:
                            errors.append(
                                "Math assurance history series is missing an entry for latestGraphVersion "
                                f"{latest_graph_version}."
                            )
                        elif version_at_least(latest_graph_version, PUBLISH_GATES_ENFORCEMENT_VERSION):
                            publish_ready = latest_series_entry.get("publishReady")
                            if not isinstance(publish_ready, bool):
                                errors.append(
                                    "Latest math assurance history entry must include boolean publishReady."
                                )
                            elif not publish_ready:
                                errors.append(
                                    "Latest math assurance history entry indicates publishReady=false."
                                )

    return errors


def validate_checksums() -> list[str]:
    errors: list[str] = []
    payload = load_json(CHECKSUMS_PATH)

    if not isinstance(payload, dict):
        return ["releases/checksums.json must be a JSON object."]

    if payload.get("algorithm") != "sha256":
        errors.append("releases/checksums.json algorithm must be 'sha256'.")

    files = payload.get("files")
    if not isinstance(files, dict):
        errors.append("releases/checksums.json 'files' must be an object mapping path to hash.")
        return errors

    expected = compute_release_checksums()
    actual = {k: v for k, v in files.items() if isinstance(k, str)}

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(path for path in set(expected) & set(actual) if expected[path] != actual[path])

    if missing:
        errors.append("checksums missing paths: " + ", ".join(missing))
    if extra:
        errors.append("checksums include unexpected paths: " + ", ".join(extra))
    if changed:
        errors.append("checksums are stale for paths: " + ", ".join(changed))

    for path, digest in files.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            errors.append(f"Invalid sha256 digest for path {path!r}.")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-checksum-validation",
        action="store_true",
        help="Validate manifest and latest pointers without checking releases/checksums.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        manifest = load_json(MANIFEST_PATH)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    errors = validate_manifest(manifest)
    errors.extend(validate_release_graph_semantics(manifest))
    errors.extend(validate_release_report_artifacts(manifest))

    if not args.skip_checksum_validation:
        try:
            errors.extend(validate_checksums())
        except RuntimeError as exc:
            errors.append(str(exc))

    if errors:
        print("Release contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Release contract validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
