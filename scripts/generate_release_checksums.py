#!/usr/bin/env python3
"""Generate or verify sha256 checksums for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASES_DIR = REPO_ROOT / "releases"
OUTPUT_PATH = RELEASES_DIR / "checksums.json"


def iter_release_artifacts() -> list[Path]:
    artifacts: list[Path] = []
    for path in RELEASES_DIR.rglob("*.json"):
        if path.resolve() == OUTPUT_PATH.resolve():
            continue
        artifacts.append(path)
    artifacts.sort()
    return artifacts


def compute_checksums() -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    for path in iter_release_artifacts():
        rel = path.relative_to(REPO_ROOT).as_posix()
        checksums[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def build_payload() -> dict:
    return {
        "algorithm": "sha256",
        "files": compute_checksums(),
    }


def load_existing_payload() -> dict | None:
    if not OUTPUT_PATH.is_file():
        return None
    try:
        return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def print_mismatch(expected: dict, actual: dict | None) -> None:
    print("releases/checksums.json is missing or out of date.", file=sys.stderr)
    if actual is None:
        print("  Existing file is missing or invalid JSON.", file=sys.stderr)
    else:
        expected_files = expected.get("files", {})
        actual_files = actual.get("files", {}) if isinstance(actual, dict) else {}
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        changed = sorted(
            path
            for path in set(expected_files) & set(actual_files)
            if expected_files[path] != actual_files[path]
        )
        if missing:
            print("  Missing entries:", file=sys.stderr)
            for path in missing:
                print(f"    - {path}", file=sys.stderr)
        if extra:
            print("  Unexpected entries:", file=sys.stderr)
            for path in extra:
                print(f"    - {path}", file=sys.stderr)
        if changed:
            print("  Changed entries:", file=sys.stderr)
            for path in changed:
                print(f"    - {path}", file=sys.stderr)


def write_payload(payload: dict) -> None:
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify releases/checksums.json without rewriting it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload()

    if args.check:
        existing = load_existing_payload()
        if existing != payload:
            print_mismatch(payload, existing)
            print("Run: python3 scripts/generate_release_checksums.py", file=sys.stderr)
            return 1
        print("Checksum verification passed.")
        return 0

    write_payload(payload)
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
