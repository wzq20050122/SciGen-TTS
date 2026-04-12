#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from cal_score import cal_score_single


METRIC_KEYS = [
    "strict_score",
    "relaxed_score",
    "semantic_correctness",
    "spelling",
    "readability",
    "logical_consistency",
]


def load_ids(path: Path) -> Set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_eval_scores(eval_dir: Path, valid_ids: Set[str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for p in sorted(eval_dir.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            row = json.load(f)
        rid = row.get("id")
        if rid in valid_ids:
            out[rid] = cal_score_single(row)
    return out


def average_scores(score_items: List[Dict]) -> Dict[str, float]:
    if not score_items:
        return {k: 0.0 for k in METRIC_KEYS}
    n = len(score_items)
    return {k: sum(x[k] for x in score_items) / n for k in METRIC_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge shared/split eval results into final overall step1/final scores."
    )
    parser.add_argument("--same_ids", type=Path, required=True, help="ids_step1_final_same.txt")
    parser.add_argument("--different_ids", type=Path, required=True, help="ids_step1_final_different.txt")

    parser.add_argument("--eval_shared_dir", type=Path, required=True, help="Shared eval dir for same ids")
    parser.add_argument("--eval_step1_diff_dir", type=Path, required=True, help="Step1 eval dir for different ids")
    parser.add_argument("--eval_final_diff_dir", type=Path, required=True, help="Final eval dir for different ids")

    parser.add_argument("--output_json", type=Path, default=None, help="Optional output json path")

    args = parser.parse_args()

    same_ids = load_ids(args.same_ids)
    diff_ids = load_ids(args.different_ids)

    if same_ids & diff_ids:
        raise RuntimeError("same_ids and different_ids overlap, which should not happen")

    shared_scores = read_eval_scores(args.eval_shared_dir, same_ids)
    step1_diff_scores = read_eval_scores(args.eval_step1_diff_dir, diff_ids)
    final_diff_scores = read_eval_scores(args.eval_final_diff_dir, diff_ids)

    missing_shared = sorted(same_ids - set(shared_scores.keys()))
    missing_step1_diff = sorted(diff_ids - set(step1_diff_scores.keys()))
    missing_final_diff = sorted(diff_ids - set(final_diff_scores.keys()))

    if missing_shared or missing_step1_diff or missing_final_diff:
        raise RuntimeError(
            "Missing eval results detected. "
            f"missing_shared={len(missing_shared)}, "
            f"missing_step1_diff={len(missing_step1_diff)}, "
            f"missing_final_diff={len(missing_final_diff)}"
        )

    combined_step1_items = [shared_scores[i] for i in sorted(same_ids)] + [step1_diff_scores[i] for i in sorted(diff_ids)]
    combined_final_items = [shared_scores[i] for i in sorted(same_ids)] + [final_diff_scores[i] for i in sorted(diff_ids)]

    step1_avg = average_scores(combined_step1_items)
    final_avg = average_scores(combined_final_items)

    result = {
        "counts": {
            "all": len(same_ids) + len(diff_ids),
            "same": len(same_ids),
            "different": len(diff_ids),
        },
        "step1_overall": step1_avg,
        "final_overall": final_avg,
        "delta_final_minus_step1": {k: final_avg[k] - step1_avg[k] for k in METRIC_KEYS},
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
