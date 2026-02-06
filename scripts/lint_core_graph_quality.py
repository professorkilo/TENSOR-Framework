#!/usr/bin/env python3
"""Validate graph quality gates and emit deterministic quality artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core_graph_pipeline import (
    ValidationError,
    build_coverage_matrix_markdown,
    build_coverage_matrix_payload,
    build_quality_report_markdown,
    load_entry_config,
    load_jsonl,
    validate_and_measure,
    write_json,
)
from core_graph_math_assurance import (
    build_math_assurance_history_markdown,
    build_math_assurance_history_payload,
    build_math_assurance_markdown,
    build_math_assurance_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="Release version in <MAJOR>.<YYYYMMDD>[REV] format (for example: 0.20260206c).",
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
        help="Path to entry config JSON, relative to repo root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for reports. Defaults to releases/core/reports/v<version>/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root: Path = args.repo_root.resolve()
    nodes_path = (repo_root / args.nodes_path).resolve()
    edges_path = (repo_root / args.edges_path).resolve()
    entry_path = (repo_root / args.entry_path).resolve()

    output_dir = (
        (repo_root / f"releases/core/reports/v{args.version}").resolve()
        if args.output_dir is None
        else (repo_root / args.output_dir).resolve()
    )

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

        output_dir.mkdir(parents=True, exist_ok=True)

        quality_json_path = output_dir / "graph-quality.json"
        quality_md_path = output_dir / "graph-quality.md"
        coverage_json_path = output_dir / "coverage-matrix.json"
        coverage_md_path = output_dir / "coverage-matrix.md"
        math_json_path = output_dir / "math-assurance.json"
        math_md_path = output_dir / "math-assurance.md"

        quality_json_payload = {
            "version": args.version,
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }
        write_json(quality_json_path, quality_json_payload)
        quality_md_path.write_text(
            build_quality_report_markdown(args.version, errors, warnings, metrics),
            encoding="utf-8",
        )

        coverage_payload = build_coverage_matrix_payload(args.version, metrics)
        write_json(coverage_json_path, coverage_payload)
        coverage_md_path.write_text(
            build_coverage_matrix_markdown(args.version, metrics),
            encoding="utf-8",
        )

        math_payload = build_math_assurance_payload(
            version=args.version,
            nodes=nodes,
            edges=edges,
            entry_node_ids=entry_node_ids,
            quality_metrics=metrics,
        )
        publish_gates = (
            math_payload.get("monitoring", {}).get("publishGates")
            if isinstance(math_payload.get("monitoring"), dict)
            else None
        )
        if not isinstance(publish_gates, dict):
            errors.append("Math assurance payload missing monitoring.publishGates.")
        else:
            all_passed = publish_gates.get("allPassed")
            failed_gate_ids = publish_gates.get("failedGateIds")
            if not isinstance(all_passed, bool):
                errors.append("Math assurance publish gates must include boolean allPassed.")
            if not isinstance(failed_gate_ids, list):
                errors.append("Math assurance publish gates must include failedGateIds list.")
                failed_gate_ids = []
            if isinstance(all_passed, bool) and not all_passed:
                rendered = ", ".join(str(gate_id) for gate_id in failed_gate_ids) or "unknown"
                errors.append(f"Publish gates failed: {rendered}")
        write_json(math_json_path, math_payload)
        math_md_path.write_text(
            build_math_assurance_markdown(math_payload),
            encoding="utf-8",
        )

        latest_reports_dir = (repo_root / "releases/core/reports/latest").resolve()
        latest_reports_dir.mkdir(parents=True, exist_ok=True)
        latest_math_json_path = latest_reports_dir / "math-assurance.json"
        latest_math_md_path = latest_reports_dir / "math-assurance.md"
        write_json(latest_math_json_path, math_payload)
        latest_math_md_path.write_text(
            build_math_assurance_markdown(math_payload),
            encoding="utf-8",
        )

        history_reports_dir = (repo_root / "releases/core/reports/history").resolve()
        history_reports_dir.mkdir(parents=True, exist_ok=True)
        history_payload = build_math_assurance_history_payload(repo_root / "releases/core/reports")
        history_json_path = history_reports_dir / "math-assurance-history.json"
        history_md_path = history_reports_dir / "math-assurance-history.md"
        write_json(history_json_path, history_payload)
        history_md_path.write_text(
            build_math_assurance_history_markdown(history_payload),
            encoding="utf-8",
        )

        print(f"Wrote {quality_json_path.relative_to(repo_root)}")
        print(f"Wrote {quality_md_path.relative_to(repo_root)}")
        print(f"Wrote {coverage_json_path.relative_to(repo_root)}")
        print(f"Wrote {coverage_md_path.relative_to(repo_root)}")
        print(f"Wrote {math_json_path.relative_to(repo_root)}")
        print(f"Wrote {math_md_path.relative_to(repo_root)}")
        print(f"Wrote {latest_math_json_path.relative_to(repo_root)}")
        print(f"Wrote {latest_math_md_path.relative_to(repo_root)}")
        print(f"Wrote {history_json_path.relative_to(repo_root)}")
        print(f"Wrote {history_md_path.relative_to(repo_root)}")

        if errors:
            print("Graph quality validation failed:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1

        if warnings:
            print("Graph quality warnings:")
            for warning in warnings:
                print(f"- {warning}")

        print("Graph quality validation passed.")
        return 0

    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
