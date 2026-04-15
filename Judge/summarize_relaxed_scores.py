import argparse
import json
import os
from glob import glob
from typing import Dict, List, Optional

from cal_score import cal_score_single


def _load_sampled_ids(sampled_id_path: Optional[str]) -> Optional[set]:
    if not sampled_id_path:
        return None
    with open(sampled_id_path, "r", encoding="utf-8") as f:
        return {x.strip() for x in f.readlines() if x.strip()}


def _collect_eval_results(eval_results_dir: str, sampled_ids: Optional[set]) -> List[Dict]:
    records = []
    required_keys = {"global_evaluation", "answers", "scoring_points"}

    for path in glob(os.path.join(eval_results_dir, "**/*.json"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        if not required_keys.issubset(record):
            continue

        if sampled_ids is not None and record.get("id") not in sampled_ids:
            continue

        records.append(record)
    return records


def summarize_step(step_name: str, eval_results_dir: str, sampled_id_path: Optional[str]) -> Dict[str, object]:
    sampled_ids = _load_sampled_ids(sampled_id_path)
    records = _collect_eval_results(eval_results_dir, sampled_ids)

    if not records:
        return {
            "step": step_name,
            "eval_results_dir": eval_results_dir,
            "count": 0,
            "relaxed_score_avg": None,
            "strict_score_avg": None,
        }

    scores = [cal_score_single(r) for r in records]
    relaxed_avg = sum(s["relaxed_score"] for s in scores) / len(scores)
    strict_avg = sum(s["strict_score"] for s in scores) / len(scores)

    return {
        "step": step_name,
        "eval_results_dir": eval_results_dir,
        "count": len(scores),
        "relaxed_score_avg": round(relaxed_avg * 100, 1),
        "strict_score_avg": round(strict_avg * 100, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize relaxed_score across step eval result directories")
    parser.add_argument("--run_tag", type=str, required=True, help="Run tag folder under Judge_output, e.g. wan2.6-...")
    parser.add_argument("--steps", type=str, default="step1,step2,step3,final", help="Comma-separated step names")
    parser.add_argument("--sampled_id_path", type=str, default=None, help="Optional id subset file")
    parser.add_argument("--output_json", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    base_dir = os.path.join("/root/autodl-tmp/TTS/Judge_output", args.run_tag)
    step_names = [x.strip() for x in args.steps.split(",") if x.strip()]

    summary = []
    for step_name in step_names:
        eval_results_dir = os.path.join(base_dir, f"eval_results_{step_name}")
        summary.append(summarize_step(step_name, eval_results_dir, args.sampled_id_path))

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
