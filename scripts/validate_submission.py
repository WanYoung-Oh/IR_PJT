#!/usr/bin/env python3
"""제출 파일(한 줄 JSON)에 필수 키가 있는지 검사."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ir_rag.competition_metrics import load_submission_rows
from ir_rag.submission import SUBMISSION_REQUIRED_KEYS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="sample_submission.csv 등")
    args = parser.parse_args()
    rows = load_submission_rows(Path(args.path))
    for i, row in enumerate(rows):
        miss = SUBMISSION_REQUIRED_KEYS - row.keys()
        if miss:
            print(f"FAIL line {i+1}: missing {sorted(miss)}")
            sys.exit(1)
    print(f"OK {len(rows)} lines; keys {sorted(SUBMISSION_REQUIRED_KEYS)}")


if __name__ == "__main__":
    main()
