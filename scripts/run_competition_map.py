#!/usr/bin/env python3
"""공개 변형 MAP(calc_map): 제출본(.csv JSONL) + GT JSONL 이 있을 때만."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ir_rag.competition_metrics import calc_map, load_gt_jsonl, load_submission_rows
from ir_rag.config import repo_root_from, resolve_config_path
from ir_rag.submission import SUBMISSION_REQUIRED_KEYS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        required=True,
        help="sample_submission.csv 형식 (한 줄 JSON)",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="eval_gt.jsonl: {eval_id, relevant_docids}",
    )
    parser.add_argument(
        "--validate-keys",
        action="store_true",
        help="각 행에 필수 키만 검사",
    )
    args = parser.parse_args()
    root = repo_root_from(Path.cwd())
    sub_path = Path(args.submission)
    if not sub_path.is_absolute():
        sub_path = root / sub_path
    gt_path = Path(args.gt)
    if not gt_path.is_absolute():
        gt_path = root / gt_path

    pred = load_submission_rows(sub_path)
    if args.validate_keys:
        for i, row in enumerate(pred):
            miss = SUBMISSION_REQUIRED_KEYS - row.keys()
            if miss:
                raise SystemExit(f"line {i+1} missing keys: {miss}")
    gt = load_gt_jsonl(gt_path)
    score = calc_map(gt, pred)
    print(f"competition MAP (calc_map): {score:.6f}  (n={len(pred)})")


if __name__ == "__main__":
    main()
