#!/usr/bin/env python3
"""Evaluate RareDx against a JSONL gold-standard benchmark.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval" / "rare_disease_cases.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "exp-results"
SOURCE = "scripts/evaluate.py"

STRATEGY_SPECS = {
    "fusion_configured": {
        "ranking_strategy": "fusion",
        "graph_weight": None,
        "semantic_weight": None,
    },
    "graph_only": {
        "ranking_strategy": "graph_only",
        "graph_weight": 1.0,
        "semantic_weight": 0.0,
    },
    "semantic_only": {
        "ranking_strategy": "semantic_only",
        "graph_weight": 0.0,
        "semantic_weight": 1.0,
    },
    "fusion_60_40": {
        "ranking_strategy": "fusion",
        "graph_weight": 0.6,
        "semantic_weight": 0.4,
    },
    "fusion_70_30": {
        "ranking_strategy": "fusion",
        "graph_weight": 0.7,
        "semantic_weight": 0.3,
    },
    "fusion_50_50": {
        "ranking_strategy": "fusion",
        "graph_weight": 0.5,
        "semantic_weight": 0.5,
    },
    "fusion_30_70": {
        "ranking_strategy": "fusion",
        "graph_weight": 0.3,
        "semantic_weight": 0.7,
    },
}

DEFAULT_STRATEGY_NAMES = ["fusion_configured"]
ABLATION_STRATEGY_NAMES = [
    "graph_only",
    "semantic_only",
    "fusion_configured",
    "fusion_60_40",
    "fusion_70_30",
    "fusion_50_50",
    "fusion_30_70",
]

REQUIRED_FIELDS = {
    "case_id",
    "clinical_note",
    "hpo_terms",
    "genes",
    "true_orphacode",
    "true_disease",
    "expected_icd10",
    "source",
    "difficulty",
}

RESULT_FIELDS = [
    "strategy",
    "ranking_strategy",
    "graph_weight",
    "semantic_weight",
    "case_id",
    "difficulty",
    "true_disease",
    "true_orphacode",
    "expected_icd10",
    "status",
    "error",
    "runtime_ms",
    "num_candidates",
    "true_rank",
    "top1_correct",
    "recall_at_3",
    "recall_at_5",
    "mrr",
    "top_disease",
    "top_orphacode",
    "top_score",
    "top_icd10",
    "top_evidence_badges",
    "graph_candidate_count",
    "citation_candidate_count",
    "icd10_candidate_count",
    "unsupported_candidate_count",
    "unsupported_candidate_drop_count",
    "unsupported_candidate_drop_rate",
    "hallucination_unsupported_output",
    "true_graph_hit",
    "true_citation_hit",
    "true_icd10_match",
    "ranked_candidates_json",
]

SUMMARY_FIELDS = [
    "scope",
    "metric_group",
    "cases",
    "top1_accuracy",
    "recall_at_3",
    "recall_at_5",
    "mrr",
    "median_true_rank",
    "no_result_rate",
    "graph_evidence_coverage",
    "pubmed_citation_coverage",
    "icd10_mapping_coverage",
    "graph_candidate_coverage",
    "citation_candidate_coverage",
    "icd10_candidate_coverage",
    "true_graph_hit_rate",
    "true_citation_hit_rate",
    "true_icd10_match_rate",
    "unsupported_candidate_count",
    "unsupported_candidate_rate",
    "unsupported_candidate_drop_count",
    "unsupported_candidate_drop_rate",
    "hallucination_unsupported_output_rate",
    "runtime_ms_mean",
    "runtime_ms_median",
    "runtime_ms_p95",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RareDx over a benchmark JSONL file and export metrics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Benchmark JSONL path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for timestamped results. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional timestamp label for deterministic output file names.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases, useful for smoke tests. Omit to run all cases.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly evaluate every case in the benchmark file.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run graph-only, semantic-only, current 60/40, and tuned fusion strategies.",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=sorted(STRATEGY_SPECS),
        help=(
            "Strategy to run. Can be supplied more than once. "
            f"Choices: {', '.join(sorted(STRATEGY_SPECS))}."
        ),
    )
    parser.add_argument(
        "--graph-weight",
        type=float,
        default=None,
        help=(
            "Override GRAPH_WEIGHT for a single fusion run. "
            "Must be supplied with --semantic-weight."
        ),
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=None,
        help=(
            "Override SEMANTIC_WEIGHT for a single fusion run. "
            "Must be supplied with --graph-weight."
        ),
    )
    parser.add_argument(
        "--weight-sweep",
        action="append",
        default=[],
        metavar="GRAPH:SEMANTIC",
        help=(
            "Run a custom fusion weight pair, e.g. 0.7:0.3. "
            "Can be supplied multiple times to sweep weights."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help="Maximum ranked candidates to serialize per case.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first case-level agent error.",
    )
    args = parser.parse_args()
    if args.all and args.limit is not None:
        parser.error("Use either --all or --limit, not both.")
    if args.ablation and args.strategy:
        parser.error("Use either --ablation or --strategy, not both.")
    has_weight_override = (
        args.graph_weight is not None or args.semantic_weight is not None
    )
    if has_weight_override and (
        args.graph_weight is None or args.semantic_weight is None
    ):
        parser.error("Use --graph-weight and --semantic-weight together.")
    if has_weight_override and (args.ablation or args.strategy or args.weight_sweep):
        parser.error(
            "Use --graph-weight/--semantic-weight without --ablation, "
            "--strategy, or --weight-sweep."
        )
    if has_weight_override:
        try:
            validate_weight_pair(args.graph_weight, args.semantic_weight)
        except ValueError as exc:
            parser.error(str(exc))
    if args.weight_sweep and (args.ablation or args.strategy):
        parser.error("Use --weight-sweep without --ablation or --strategy.")
    for weight_pair in args.weight_sweep:
        try:
            parse_weight_sweep_spec(weight_pair)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def load_cases(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc

            validate_case(case, line_number)
            case_id = case["case_id"]
            if case_id in case_ids:
                raise ValueError(f"Duplicate case_id at line {line_number}: {case_id}")
            case_ids.add(case_id)
            cases.append(case)

            if limit is not None and len(cases) >= limit:
                break

    if not cases:
        raise ValueError(f"No benchmark cases found in {path}")

    return cases


def validate_case(case: dict[str, Any], line_number: int) -> None:
    missing = REQUIRED_FIELDS - set(case)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Line {line_number} is missing required fields: {missing_text}")

    if not isinstance(case["case_id"], str) or not case["case_id"].strip():
        raise ValueError(f"Line {line_number} has invalid case_id")
    if not isinstance(case["clinical_note"], str) or not case["clinical_note"].strip():
        raise ValueError(f"Line {line_number} has invalid clinical_note")
    if not isinstance(case["hpo_terms"], list) or not all(
        isinstance(item, str) for item in case["hpo_terms"]
    ):
        raise ValueError(f"Line {line_number} has invalid hpo_terms")
    if not isinstance(case["genes"], list) or not all(
        isinstance(item, str) for item in case["genes"]
    ):
        raise ValueError(f"Line {line_number} has invalid genes")
    if case["difficulty"] not in {"easy", "medium", "hard"}:
        raise ValueError(f"Line {line_number} has invalid difficulty")


def selected_strategy_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.weight_sweep:
        return [parse_weight_sweep_spec(item) for item in args.weight_sweep]

    if args.graph_weight is not None:
        return [
            custom_fusion_spec(
                name=weight_strategy_name(args.graph_weight, args.semantic_weight),
                graph_weight=args.graph_weight,
                semantic_weight=args.semantic_weight,
            )
        ]

    if args.ablation:
        names = ABLATION_STRATEGY_NAMES
    elif args.strategy:
        names = list(dict.fromkeys(args.strategy))
    else:
        names = DEFAULT_STRATEGY_NAMES

    return [strategy_spec(name) for name in names]


def strategy_spec(name: str) -> dict[str, Any]:
    if name == "fusion_configured":
        from core.config import config

        return custom_fusion_spec(
            name="fusion_configured",
            graph_weight=config.graph_weight,
            semantic_weight=config.semantic_weight,
        )

    spec = dict(STRATEGY_SPECS[name])
    spec["name"] = name
    validate_weight_pair(spec["graph_weight"], spec["semantic_weight"])
    return spec


def parse_weight_sweep_spec(value: str) -> dict[str, Any]:
    parts = re.split(r"[:,/]", value)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --weight-sweep value '{value}'. Expected GRAPH:SEMANTIC."
        )

    try:
        graph_weight = float(parts[0])
        semantic_weight = float(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Invalid --weight-sweep value '{value}'. Weights must be numbers."
        ) from exc

    return custom_fusion_spec(
        name=weight_strategy_name(graph_weight, semantic_weight),
        graph_weight=graph_weight,
        semantic_weight=semantic_weight,
    )


def custom_fusion_spec(
    name: str,
    graph_weight: float,
    semantic_weight: float,
) -> dict[str, Any]:
    validate_weight_pair(graph_weight, semantic_weight)
    return {
        "name": name,
        "ranking_strategy": "fusion",
        "graph_weight": graph_weight,
        "semantic_weight": semantic_weight,
    }


def validate_weight_pair(graph_weight: float, semantic_weight: float) -> None:
    if graph_weight < 0.0 or graph_weight > 1.0:
        raise ValueError("graph_weight must be between 0.0 and 1.0")
    if semantic_weight < 0.0 or semantic_weight > 1.0:
        raise ValueError("semantic_weight must be between 0.0 and 1.0")
    if abs((graph_weight + semantic_weight) - 1.0) > 1e-6:
        raise ValueError("graph_weight and semantic_weight must sum to 1.0")


def weight_strategy_name(graph_weight: float, semantic_weight: float) -> str:
    return f"fusion_{weight_label(graph_weight)}_{weight_label(semantic_weight)}"


def weight_label(weight: float) -> str:
    percentage = weight * 100
    if percentage.is_integer():
        return str(int(percentage))
    return f"{percentage:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def build_agent_state(case: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "clinical_note": case["clinical_note"],
        "selected_symptoms": case.get("selected_symptoms", case["hpo_terms"]),
        "selected_genes": case.get("selected_genes", case["genes"]),
        "patient_id": case["case_id"],
        "patient_age": case.get("patient_age", 0),
        "patient_gender": case.get("patient_gender", "Unknown"),
        "graph_results": [],
        "semantic_results": [],
        "candidates": [],
        "validated_evidence": [],
        "final_report": {},
        "audit_trail": [],
        "ranking_strategy": strategy["ranking_strategy"],
        "graph_weight": strategy["graph_weight"],
        "semantic_weight": strategy["semantic_weight"],
    }


def evaluate_cases(
    cases: list[dict[str, Any]],
    strategies: list[dict[str, Any]],
    max_candidates: int,
    fail_fast: bool,
) -> list[dict[str, Any]]:
    from core.agent import rare_dx_agent

    rows: list[dict[str, Any]] = []
    total = len(cases)
    strategy_total = len(strategies)

    for index, case in enumerate(cases, start=1):
        for strategy_index, strategy in enumerate(strategies, start=1):
            print(
                f"[{index}/{total}][{strategy_index}/{strategy_total}] "
                f"Evaluating {case['case_id']} with {strategy['name']}...",
                flush=True,
            )
            started = time.perf_counter()

            try:
                final_state = rare_dx_agent.invoke(build_agent_state(case, strategy))
                runtime_ms = elapsed_ms(started)
                row = build_result_row(
                    case,
                    strategy,
                    final_state,
                    runtime_ms,
                    max_candidates,
                )
            except Exception as exc:  # noqa: BLE001 - evaluation should capture case errors.
                runtime_ms = elapsed_ms(started)
                row = build_error_row(case, strategy, runtime_ms, exc)
                if fail_fast:
                    rows.append(row)
                    raise

            rows.append(row)

    return rows


def build_result_row(
    case: dict[str, Any],
    strategy: dict[str, Any],
    final_state: dict[str, Any],
    runtime_ms: int,
    max_candidates: int,
) -> dict[str, Any]:
    candidates = final_state.get("candidates", []) or []
    candidate_payload = summarize_candidates(candidates, max_candidates)
    true_rank = find_true_rank(case, candidates)
    true_candidate = candidates[true_rank - 1] if true_rank else None
    top_candidate = candidates[0] if candidates else {}
    graph_candidate_count = sum(1 for candidate in candidates if has_graph_evidence(candidate))
    citation_candidate_count = sum(
        1 for candidate in candidates if has_citation_evidence(candidate)
    )
    icd10_candidate_count = sum(1 for candidate in candidates if candidate.get("icd10"))
    unsupported_candidate_count = sum(
        1 for candidate in candidates if is_unsupported_candidate(candidate)
    )
    unsupported_candidate_drop_count = extract_unsupported_drop_count(final_state)
    unsupported_candidate_drop_rate = rate(
        unsupported_candidate_drop_count,
        len(candidates) + unsupported_candidate_drop_count,
    )

    return {
        "strategy": strategy["name"],
        "ranking_strategy": strategy["ranking_strategy"],
        "graph_weight": strategy["graph_weight"],
        "semantic_weight": strategy["semantic_weight"],
        "case_id": case["case_id"],
        "difficulty": case["difficulty"],
        "true_disease": case["true_disease"],
        "true_orphacode": case["true_orphacode"],
        "expected_icd10": case["expected_icd10"],
        "status": "ok",
        "error": "",
        "runtime_ms": runtime_ms,
        "num_candidates": len(candidates),
        "true_rank": true_rank or "",
        "top1_correct": int(true_rank == 1),
        "recall_at_3": int(true_rank is not None and true_rank <= 3),
        "recall_at_5": int(true_rank is not None and true_rank <= 5),
        "mrr": round(1 / true_rank, 6) if true_rank else 0.0,
        "top_disease": top_candidate.get("disease", ""),
        "top_orphacode": format_orphacode(top_candidate.get("orphacode")),
        "top_score": round(float(top_candidate.get("score", 0.0)), 6)
        if top_candidate
        else "",
        "top_icd10": top_candidate.get("icd10", ""),
        "top_evidence_badges": json.dumps(
            top_candidate.get("evidence_badges", []), sort_keys=True
        ),
        "graph_candidate_count": graph_candidate_count,
        "citation_candidate_count": citation_candidate_count,
        "icd10_candidate_count": icd10_candidate_count,
        "unsupported_candidate_count": unsupported_candidate_count,
        "unsupported_candidate_drop_count": unsupported_candidate_drop_count,
        "unsupported_candidate_drop_rate": unsupported_candidate_drop_rate,
        "hallucination_unsupported_output": int(unsupported_candidate_count > 0),
        "true_graph_hit": int(bool(true_candidate and has_graph_evidence(true_candidate))),
        "true_citation_hit": int(bool(true_candidate and has_citation_evidence(true_candidate))),
        "true_icd10_match": int(
            bool(
                true_candidate
                and normalize_icd10(true_candidate.get("icd10"))
                == normalize_icd10(case["expected_icd10"])
            )
        ),
        "ranked_candidates_json": json.dumps(candidate_payload, sort_keys=True),
    }


def build_error_row(
    case: dict[str, Any],
    strategy: dict[str, Any],
    runtime_ms: int,
    exc: Exception,
) -> dict[str, Any]:
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "strategy": strategy["name"],
            "ranking_strategy": strategy["ranking_strategy"],
            "graph_weight": strategy["graph_weight"],
            "semantic_weight": strategy["semantic_weight"],
            "case_id": case["case_id"],
            "difficulty": case["difficulty"],
            "true_disease": case["true_disease"],
            "true_orphacode": case["true_orphacode"],
            "expected_icd10": case["expected_icd10"],
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_ms": runtime_ms,
            "num_candidates": 0,
            "top1_correct": 0,
            "recall_at_3": 0,
            "recall_at_5": 0,
            "mrr": 0.0,
            "graph_candidate_count": 0,
            "citation_candidate_count": 0,
            "icd10_candidate_count": 0,
            "unsupported_candidate_count": 0,
            "unsupported_candidate_drop_count": 0,
            "unsupported_candidate_drop_rate": 0.0,
            "hallucination_unsupported_output": 0,
            "true_graph_hit": 0,
            "true_citation_hit": 0,
            "true_icd10_match": 0,
            "ranked_candidates_json": "[]",
        }
    )
    return row


def summarize_candidates(
    candidates: list[dict[str, Any]], max_candidates: int
) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates[:max_candidates], start=1):
        citations = candidate.get("citations", []) or []
        summarized.append(
            {
                "rank": rank,
                "disease": candidate.get("disease"),
                "orphacode": format_orphacode(candidate.get("orphacode")),
                "score": round(float(candidate.get("score", 0.0)), 6),
                "icd10": candidate.get("icd10"),
                "evidence_badges": candidate.get("evidence_badges", []),
                "graph_path_count": len(candidate.get("graph_paths", []) or []),
                "citation_count": len(citations),
                "unsupported": is_unsupported_candidate(candidate),
                "pmids": [
                    article.get("pmid")
                    for article in citations
                    if isinstance(article, dict) and article.get("pmid")
                ],
                "matched_symptoms": candidate.get("matched_symptoms", []),
                "matched_genes": candidate.get("matched_genes", []),
            }
        )
    return summarized


def has_graph_evidence(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("graph_paths"))


def has_citation_evidence(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("citations"))


def is_unsupported_candidate(candidate: dict[str, Any]) -> bool:
    return not has_graph_evidence(candidate) and not has_citation_evidence(candidate)


def extract_unsupported_drop_count(final_state: dict[str, Any]) -> int:
    """Return the number of candidates removed by the fusion evidence guard."""
    for event in final_state.get("audit_trail", []) or []:
        if event.get("node") != "FusionGuard":
            continue
        match = re.search(r"Dropped\s+(\d+)", event.get("details", ""))
        if match:
            return int(match.group(1))

    pre_fusion_count = count_unique_prefusion_candidates(final_state)
    returned_count = len(final_state.get("candidates", []) or [])
    return max(pre_fusion_count - returned_count, 0)


def count_unique_prefusion_candidates(final_state: dict[str, Any]) -> int:
    keys = set()
    for source_key in ["graph_results", "semantic_results"]:
        for candidate in final_state.get(source_key, []) or []:
            key = candidate_identity_key(candidate)
            if key:
                keys.add(key)
    return len(keys)


def candidate_identity_key(candidate: dict[str, Any]) -> str:
    orphacode = normalize_orphacode(candidate.get("orphacode"))
    if orphacode:
        return f"orpha:{orphacode}"
    name = normalize_name(candidate.get("disease"))
    return f"name:{name}" if name else ""


def find_true_rank(case: dict[str, Any], candidates: list[dict[str, Any]]) -> int | None:
    true_orphacode = normalize_orphacode(case["true_orphacode"])
    true_name = normalize_name(case["true_disease"])

    for rank, candidate in enumerate(candidates, start=1):
        candidate_orphacode = normalize_orphacode(candidate.get("orphacode"))
        candidate_name = normalize_name(candidate.get("disease"))
        if true_orphacode and candidate_orphacode == true_orphacode:
            return rank
        if true_name and candidate_name == true_name:
            return rank
    return None


def build_summary(rows: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    error_rows = [row for row in rows if row["status"] != "ok"]

    summary: dict[str, Any] = {
        "source": SOURCE,
        "input_path": str(input_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_cases": len(rows),
        "unique_cases": len({row["case_id"] for row in rows}),
        "strategy_count": len({row["strategy"] for row in rows}),
        "evaluated_cases": len(ok_rows),
        "failed_cases": len(error_rows),
        "error_rate": rate(len(error_rows), len(rows)),
    }

    summary.update(metric_block(ok_rows))

    by_difficulty: dict[str, Any] = {}
    for difficulty in ["easy", "medium", "hard"]:
        difficulty_rows = [row for row in ok_rows if row["difficulty"] == difficulty]
        by_difficulty[difficulty] = metric_block(difficulty_rows)
    summary["by_difficulty"] = by_difficulty

    by_strategy: dict[str, Any] = {}
    for strategy in sorted({row["strategy"] for row in ok_rows}):
        strategy_rows = [row for row in ok_rows if row["strategy"] == strategy]
        by_strategy[strategy] = metric_block(strategy_rows)
    summary["by_strategy"] = by_strategy
    summary["best_strategy"] = choose_best_strategy(by_strategy)

    status_counts = Counter(row["status"] for row in rows)
    summary["status_counts"] = dict(status_counts)
    return summary


def choose_best_strategy(by_strategy: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not by_strategy:
        return {}

    def score(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
        median_rank = float(metrics["median_true_rank"] or 0)
        median_rank_score = -median_rank if median_rank else float("-inf")
        return (
            float(metrics["top1_accuracy"]),
            float(metrics["recall_at_5"]),
            float(metrics["mrr"]),
            median_rank_score,
            -float(metrics["hallucination_unsupported_output_rate"]),
        )

    best_score = max(score(metrics) for metrics in by_strategy.values())
    tied_strategies = [
        name for name, metrics in sorted(by_strategy.items()) if score(metrics) == best_score
    ]
    tie_preference = [
        "fusion_configured",
        "fusion_60_40",
        "fusion_70_30",
        "fusion_50_50",
        "fusion_30_70",
        "graph_only",
        "semantic_only",
    ]
    name = next(
        (candidate for candidate in tie_preference if candidate in tied_strategies),
        tied_strategies[0],
    )
    metrics = by_strategy[name]
    return {
        "strategy": name,
        "tied_strategies": tied_strategies,
        "selection_rule": (
            "highest top1_accuracy, then recall_at_5, then mrr, then lower "
            "median_true_rank; ties prefer configured fusion for continuity"
        ),
        "top1_accuracy": metrics["top1_accuracy"],
        "recall_at_5": metrics["recall_at_5"],
        "mrr": metrics["mrr"],
        "median_true_rank": metrics["median_true_rank"],
        "hallucination_unsupported_output_rate": metrics[
            "hallucination_unsupported_output_rate"
        ],
    }


def metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        empty = {
            "cases": 0,
            "top1_accuracy": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "median_true_rank": 0,
            "no_result_rate": 0.0,
            "graph_evidence_coverage": 0.0,
            "pubmed_citation_coverage": 0.0,
            "icd10_mapping_coverage": 0.0,
            "graph_candidate_coverage": 0.0,
            "citation_candidate_coverage": 0.0,
            "icd10_candidate_coverage": 0.0,
            "true_graph_hit_rate": 0.0,
            "true_citation_hit_rate": 0.0,
            "true_icd10_match_rate": 0.0,
            "unsupported_candidate_count": 0,
            "unsupported_candidate_rate": 0.0,
            "unsupported_candidate_drop_count": 0,
            "unsupported_candidate_drop_rate": 0.0,
            "hallucination_unsupported_output_rate": 0.0,
            "runtime_ms_mean": 0,
            "runtime_ms_median": 0,
            "runtime_ms_p95": 0,
        }
        return add_metric_groups(empty)

    runtimes = [int(row["runtime_ms"]) for row in rows]
    true_ranks = [int(row["true_rank"]) for row in rows if row["true_rank"]]
    total_candidates = sum(int(row["num_candidates"]) for row in rows)
    graph_candidates = sum_int(rows, "graph_candidate_count")
    citation_candidates = sum_int(rows, "citation_candidate_count")
    icd10_candidates = sum_int(rows, "icd10_candidate_count")
    unsupported_candidates = sum_int(rows, "unsupported_candidate_count")
    dropped_candidates = sum_int(rows, "unsupported_candidate_drop_count")
    raw_candidate_count = total_candidates + dropped_candidates
    metrics = {
        "cases": len(rows),
        "top1_accuracy": rate(sum_int(rows, "top1_correct"), len(rows)),
        "recall_at_3": rate(sum_int(rows, "recall_at_3"), len(rows)),
        "recall_at_5": rate(sum_int(rows, "recall_at_5"), len(rows)),
        "mrr": round(mean(float(row["mrr"]) for row in rows), 6),
        "median_true_rank": round(median(true_ranks), 3) if true_ranks else 0,
        "no_result_rate": rate(sum(1 for row in rows if int(row["num_candidates"]) == 0), len(rows)),
        "graph_evidence_coverage": rate(graph_candidates, total_candidates),
        "pubmed_citation_coverage": rate(citation_candidates, total_candidates),
        "icd10_mapping_coverage": rate(icd10_candidates, total_candidates),
        "graph_candidate_coverage": rate(
            sum(1 for row in rows if int(row["graph_candidate_count"]) > 0), len(rows)
        ),
        "citation_candidate_coverage": rate(
            sum(1 for row in rows if int(row["citation_candidate_count"]) > 0), len(rows)
        ),
        "icd10_candidate_coverage": rate(
            sum(1 for row in rows if int(row["icd10_candidate_count"]) > 0), len(rows)
        ),
        "true_graph_hit_rate": rate(sum_int(rows, "true_graph_hit"), len(rows)),
        "true_citation_hit_rate": rate(sum_int(rows, "true_citation_hit"), len(rows)),
        "true_icd10_match_rate": rate(sum_int(rows, "true_icd10_match"), len(rows)),
        "unsupported_candidate_count": unsupported_candidates,
        "unsupported_candidate_rate": rate(unsupported_candidates, total_candidates),
        "unsupported_candidate_drop_count": dropped_candidates,
        "unsupported_candidate_drop_rate": rate(dropped_candidates, raw_candidate_count),
        "hallucination_unsupported_output_rate": rate(
            sum_int(rows, "hallucination_unsupported_output"), len(rows)
        ),
        "runtime_ms_mean": round(mean(runtimes)),
        "runtime_ms_median": round(median(runtimes)),
        "runtime_ms_p95": percentile(runtimes, 95),
    }
    return add_metric_groups(metrics)


def add_metric_groups(metrics: dict[str, Any]) -> dict[str, Any]:
    metrics["diagnostic_ranking"] = {
        "cases": metrics["cases"],
        "top1_accuracy": metrics["top1_accuracy"],
        "recall_at_3": metrics["recall_at_3"],
        "recall_at_5": metrics["recall_at_5"],
        "mrr": metrics["mrr"],
        "median_true_rank": metrics["median_true_rank"],
        "no_result_rate": metrics["no_result_rate"],
    }
    metrics["evidence_safety"] = {
        "cases": metrics["cases"],
        "graph_evidence_coverage": metrics["graph_evidence_coverage"],
        "pubmed_citation_coverage": metrics["pubmed_citation_coverage"],
        "icd10_mapping_coverage": metrics["icd10_mapping_coverage"],
        "graph_candidate_coverage": metrics["graph_candidate_coverage"],
        "citation_candidate_coverage": metrics["citation_candidate_coverage"],
        "icd10_candidate_coverage": metrics["icd10_candidate_coverage"],
        "true_graph_hit_rate": metrics["true_graph_hit_rate"],
        "true_citation_hit_rate": metrics["true_citation_hit_rate"],
        "true_icd10_match_rate": metrics["true_icd10_match_rate"],
        "unsupported_candidate_count": metrics["unsupported_candidate_count"],
        "unsupported_candidate_rate": metrics["unsupported_candidate_rate"],
        "unsupported_candidate_drop_count": metrics["unsupported_candidate_drop_count"],
        "unsupported_candidate_drop_rate": metrics["unsupported_candidate_drop_rate"],
        "hallucination_unsupported_output_rate": metrics[
            "hallucination_unsupported_output_rate"
        ],
    }
    return metrics


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_summary_csv(
    summary: dict[str, Any],
    output_path: Path,
    result_rows: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = [row for row in result_rows if row["status"] == "ok"]
    rows = summary_group_rows("overall", summary)
    for strategy, metrics in summary["by_strategy"].items():
        rows.extend(summary_group_rows(f"strategy:{strategy}", metrics))
        for difficulty in ["easy", "medium", "hard"]:
            strategy_difficulty_rows = [
                row
                for row in ok_rows
                if row["strategy"] == strategy and row["difficulty"] == difficulty
            ]
            if strategy_difficulty_rows:
                rows.extend(
                    summary_group_rows(
                        f"strategy:{strategy}/difficulty:{difficulty}",
                        metric_block(strategy_difficulty_rows),
                    )
                )
    for difficulty in ["easy", "medium", "hard"]:
        rows.extend(
            summary_group_rows(
                f"difficulty:{difficulty}",
                summary["by_difficulty"][difficulty],
            )
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summary_group_rows(scope: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        summary_row(scope, "diagnostic_ranking", metrics["diagnostic_ranking"]),
        summary_row(scope, "evidence_safety", metrics["evidence_safety"]),
    ]


def summary_row(scope: str, metric_group: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row["scope"] = scope
    row["metric_group"] = metric_group
    for field in SUMMARY_FIELDS:
        if field in {"scope", "metric_group"}:
            continue
        row[field] = metrics.get(field, "")
    return row


def print_summary(
    summary: dict[str, Any],
    results_path: Path,
    summary_path: Path,
    summary_csv_path: Path,
) -> None:
    print("\nRareDx Evaluation Summary")
    print("=========================")
    if summary.get("strategy_count", 1) > 1:
        print(
            f"Evaluations: {summary['evaluated_cases']} completed / "
            f"{summary['total_cases']} strategy-case runs "
            f"({summary['unique_cases']} cases x {summary['strategy_count']} strategies)"
        )
    else:
        print(f"Cases: {summary['evaluated_cases']} evaluated / {summary['total_cases']} total")
    print(f"Failures: {summary['failed_cases']} ({summary['error_rate']:.3f})")

    print("\nDiagnostic Ranking")
    print("------------------")
    print(f"Top-1 accuracy: {summary['top1_accuracy']:.3f}")
    print(f"Recall@3: {summary['recall_at_3']:.3f}")
    print(f"Recall@5: {summary['recall_at_5']:.3f}")
    print(f"MRR: {summary['mrr']:.3f}")
    print(f"Median true rank: {summary['median_true_rank']}")
    print(f"No-result rate: {summary['no_result_rate']:.3f}")

    print("\nEvidence And Safety")
    print("-------------------")
    print(f"Graph evidence coverage: {summary['graph_evidence_coverage']:.3f}")
    print(f"PubMed citation coverage: {summary['pubmed_citation_coverage']:.3f}")
    print(f"ICD-10 mapping coverage: {summary['icd10_mapping_coverage']:.3f}")
    print(f"Unsupported candidate drop rate: {summary['unsupported_candidate_drop_rate']:.3f}")
    print(
        "Hallucination/unsupported-output rate: "
        f"{summary['hallucination_unsupported_output_rate']:.3f}"
    )

    if summary.get("by_strategy"):
        print("\nAblation Strategies")
        print("-------------------")
        for strategy, metrics in summary["by_strategy"].items():
            print(
                f"{strategy}: Top-1={metrics['top1_accuracy']:.3f}, "
                f"Recall@5={metrics['recall_at_5']:.3f}, "
                f"MRR={metrics['mrr']:.3f}, "
                f"MedianRank={metrics['median_true_rank']}"
            )
        best_strategy = summary.get("best_strategy", {})
        if best_strategy:
            tie_note = ""
            tied = best_strategy.get("tied_strategies", [])
            if len(tied) > 1:
                other_ties = [
                    strategy for strategy in tied if strategy != best_strategy["strategy"]
                ]
                tie_note = f" tied with {', '.join(other_ties)}"
            print(
                "Best strategy: "
                f"{best_strategy['strategy']} "
                f"(Top-1={best_strategy['top1_accuracy']:.3f}, "
                f"Recall@5={best_strategy['recall_at_5']:.3f}, "
                f"MRR={best_strategy['mrr']:.3f})"
                f"{tie_note}"
            )

    print("\nRuntime")
    print("-------")
    print(f"Runtime p50/p95 ms: {summary['runtime_ms_median']} / {summary['runtime_ms_p95']}")
    print(f"Results CSV: {results_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary CSV: {summary_csv_path}")


def elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def format_orphacode(value: Any) -> str:
    normalized = normalize_orphacode(value)
    return f"ORPHA:{normalized}" if normalized else ""


def normalize_orphacode(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("ORPHA:"):
        text = text.split(":", 1)[1]
    return re.sub(r"[^0-9]", "", text)


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def normalize_icd10(value: Any) -> str:
    return re.sub(r"[^A-Z0-9.]+", "", str(value or "").upper())


def sum_int(rows: list[dict[str, Any]], field: str) -> int:
    return sum(int(row[field]) for row in rows)


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def percentile(values: list[int], percentile_value: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = round((percentile_value / 100) * (len(ordered) - 1))
    return ordered[rank]


def main() -> int:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    results_path = output_dir / f"results_{timestamp}.csv"
    summary_path = output_dir / f"summary_{timestamp}.json"

    cases = load_cases(input_path, args.limit)
    strategies = selected_strategy_specs(args)
    print(f"Loaded {len(cases)} benchmark case(s) from {input_path}")
    print(f"Strategies: {', '.join(strategy['name'] for strategy in strategies)}")
    rows = evaluate_cases(cases, strategies, args.max_candidates, args.fail_fast)
    summary = build_summary(rows, input_path)

    write_results(rows, results_path)
    write_summary(summary, summary_path)
    summary_csv_path = output_dir / f"summary_{timestamp}.csv"
    write_summary_csv(summary, summary_csv_path, rows)
    print_summary(summary, results_path, summary_path, summary_csv_path)
    return 0 if summary["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
