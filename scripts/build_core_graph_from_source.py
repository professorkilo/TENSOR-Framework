#!/usr/bin/env python3
"""Build canonical TENSOR Core graph artifacts from JSONL sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core_graph_pipeline import (
    ValidationError,
    build_graph_payload,
    canonical_generated_at,
    load_entry_config,
    load_jsonl,
    validate_and_measure,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="Release version in <MAJOR>.<YYYYMMDD>[REV] format (for example: 0.20260206c).",
    )
    parser.add_argument(
        "--draft-only",
        action="store_true",
        help="Write only drafts/core/tensor.core.graph.json (do not touch releases/).",
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parent.parent,
        type=Path,
        help="Repository root path.",
    )
    parser.add_argument(
        "--nodes-path",
        default=Path("drafts/core/source/nodes.jsonl"),
        type=Path,
        help="Path to nodes JSONL, relative to repo root.",
    )
    parser.add_argument(
        "--edges-path",
        default=Path("drafts/core/source/edges.jsonl"),
        type=Path,
        help="Path to edges JSONL, relative to repo root.",
    )
    parser.add_argument(
        "--entry-path",
        default=Path("drafts/core/source/entry_nodes.json"),
        type=Path,
        help="Path to entry node config JSON, relative to repo root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root: Path = args.repo_root.resolve()
    nodes_path = (repo_root / args.nodes_path).resolve()
    edges_path = (repo_root / args.edges_path).resolve()
    entry_path = (repo_root / args.entry_path).resolve()

    try:
        nodes = load_jsonl(nodes_path)
        edges = load_jsonl(edges_path)
        entry_node_ids, fanin_exception_ids = load_entry_config(entry_path)

        errors, warnings, metrics = validate_and_measure(
            nodes=nodes,
            edges=edges,
            entry_node_ids=entry_node_ids,
            fanin_exception_node_ids=fanin_exception_ids,
        )

        if errors:
            print("Core graph build failed validation:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1

        if warnings:
            print("Core graph build warnings:")
            for warning in warnings:
                print(f"- {warning}")

        payload = build_graph_payload(
            version=args.version,
            nodes=nodes,
            edges=edges,
            entry_node_ids=entry_node_ids,
        )

        draft_graph_path = repo_root / "drafts/core/tensor.core.graph.json"

        write_json(draft_graph_path, payload)
        if not args.draft_only:
            release_graph_path = repo_root / f"releases/core/graphs/v{args.version}/tensor.core.graph.json"
            latest_graph_path = repo_root / "releases/core/graphs/latest/tensor.core.graph.json"
            write_json(release_graph_path, payload)
            write_json(latest_graph_path, payload)

        print(f"Built graph version: {args.version}")
        print(f"Generated at: {canonical_generated_at(args.version)}")
        print(f"Nodes: {metrics['nodeCount']} | Edges: {metrics['edgeCount']}")
        print(f"Entry nodes: {metrics['entryCount']} | Cross-domain ratio: {metrics['crossDomainRatio']:.4f}")
        print(f"Draft output: {draft_graph_path.relative_to(repo_root)}")
        if not args.draft_only:
            print(f"Release output: {release_graph_path.relative_to(repo_root)}")
            print(f"Latest output: {latest_graph_path.relative_to(repo_root)}")
        return 0

    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
