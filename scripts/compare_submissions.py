#!/usr/bin/env python3
"""두 제출 CSV(한 줄 JSON)를 eval_id 기준으로 비교한다.

topk doc_id 리스트·standalone_query 차이를 요약하고,
집합이 다른 경우 겹치는 doc 개수(0/1/2) 분포를 함께 출력한다.

사용 예::

    python scripts/compare_submissions.py \\
      artifacts/sample_submission_a.csv artifacts/sample_submission_b.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_rows(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        eid = int(o["eval_id"])
        rows[eid] = {
            "topk": o.get("topk") or [],
            "standalone": o.get("standalone_query", ""),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path_a", type=Path, help="첫 번째 제출 CSV")
    parser.add_argument("path_b", type=Path, help="두 번째 제출 CSV")
    args = parser.parse_args()

    if not args.path_a.is_file():
        print(f"파일 없음: {args.path_a}", file=sys.stderr)
        sys.exit(1)
    if not args.path_b.is_file():
        print(f"파일 없음: {args.path_b}", file=sys.stderr)
        sys.exit(1)

    a = load_rows(args.path_a)
    b = load_rows(args.path_b)

    print(f"A: {args.path_a}  ({len(a)}건)")
    print(f"B: {args.path_b}  ({len(b)}건)")
    print(f"eval_id 집합 동일: {set(a.keys()) == set(b.keys())}")
    if set(a.keys()) != set(b.keys()):
        only_a = set(a.keys()) - set(b.keys())
        only_b = set(b.keys()) - set(a.keys())
        if only_a:
            head = sorted(only_a)[:20]
            tail = "..." if len(only_a) > 20 else ""
            print(f"  A에만 있는 eval_id: {head}{tail}")
        if only_b:
            head = sorted(only_b)[:20]
            tail = "..." if len(only_b) > 20 else ""
            print(f"  B에만 있는 eval_id: {head}{tail}")
        sys.exit(1)

    identical = 0
    same_set = 0
    order_only: list[int] = []
    overlap_by_count: Counter[int] = Counter()
    standalone_mismatch: list[int] = []

    for eid in sorted(a.keys()):
        ta, tb = a[eid]["topk"], b[eid]["topk"]
        if a[eid]["standalone"] != b[eid]["standalone"]:
            standalone_mismatch.append(eid)

        if ta == tb:
            identical += 1
            same_set += 1
        elif set(ta) == set(tb):
            same_set += 1
            order_only.append(eid)
        else:
            overlap_by_count[len(set(ta) & set(tb))] += 1

    real_diff = sum(overlap_by_count.values())

    print()
    print("=== topk(doc_id) ===")
    print(f"완전 동일 (순서까지):        {identical}")
    print(f"집합 동일 (순서만 다름 포함): {same_set}")
    print(f"  └ 순서만 다른 건수:       {len(order_only)}")
    print(f"  └ 집합도 다른 건수:       {real_diff}")

    if real_diff:
        ovs: list[int] = []
        jacs: list[float] = []
        for eid in sorted(a.keys()):
            ta, tb = a[eid]["topk"], b[eid]["topk"]
            if set(ta) == set(tb):
                continue
            ovs.append(sum(1 for x, y in zip(ta, tb) if x == y))
            sa, sb = set(ta), set(tb)
            jacs.append(len(sa & sb) / max(len(sa | sb), 1))
        print(
            f"집합 다름 — 위치 일치 평균: {sum(ovs) / len(ovs):.2f}/3  "
            f"Jaccard 평균: {sum(jacs) / len(jacs):.3f}"
        )
        print()
        print("=== 집합이 다른 경우: 겹치는 doc 개수 분포 ===")
        for k in sorted(overlap_by_count.keys()):
            label = {
                0: "세 문서 모두 다름",
                1: "1개 동일",
                2: "2개 동일",
            }.get(k, f"{k}개 동일")
            print(f"  겹치는 doc {k}개 ({label}): {overlap_by_count[k]}건")

    print()
    print(f"standalone_query 문자열 다른 eval_id: {len(standalone_mismatch)}건")
    if standalone_mismatch:
        s = standalone_mismatch
        tail = "..." if len(s) > 60 else ""
        print(f"  {s[:60]}{tail}")


if __name__ == "__main__":
    main()
