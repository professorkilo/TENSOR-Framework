#!/usr/bin/env python3
"""Validate TENSOR release contract integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
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
