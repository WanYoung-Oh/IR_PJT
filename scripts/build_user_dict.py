#!/usr/bin/env python3
"""Build Nori user_dictionary_rules / user_dict.txt from corpus via MeCab."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.io_util import iter_jsonl

# 코퍼스 빈도와 무관하게 항상 사전에 포함할 고정 시드 용어
# 형식: (단어, 품사)  NNG=일반명사, NNP=고유명사, SL=외래어
SEED_TERMS: list[tuple[str, str]] = [
    # 여성 생식기 관련
    ("자궁", "NNG"), ("난관", "NNG"), ("나팔관", "NNG"),
    ("자궁경부", "NNG"), ("자궁내막", "NNG"), ("자궁근종", "NNG"),
    ("난소", "NNG"), ("배란", "NNG"), ("수정란", "NNG"),
    # 의학 일반
    ("의학", "NNG"), ("임상", "NNG"), ("진단", "NNG"),
    ("처방", "NNG"), ("투약", "NNG"), ("수술", "NNG"),
    ("합병증", "NNG"), ("예후", "NNG"), ("병리", "NNG"),
    # 사회·경제
    ("퇴직", "NNG"), ("퇴직금", "NNG"), ("연금", "NNG"),
    ("친환경", "NNG"), ("재생에너지", "NNG"), ("탄소중립", "NNG"),
    ("테러", "NNG"), ("테러리즘", "NNG"), ("테러리스트", "NNG"),
    ("여행", "NNG"), ("관광", "NNG"), ("여행지", "NNG"),
    ("공교육", "NNG"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()
    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    doc_path = root / cfg["paths"]["documents"]
    out_path = root / cfg["paths"]["user_dict_out"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import MeCab

    tagger = MeCab.Tagger()
    noun_counter: Counter[str] = Counter()
    mixed_counter: Counter[str] = Counter()
    mixed_pattern = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]{1,}\b")

    for doc in iter_jsonl(doc_path):
        text = doc["content"]
        parsed = tagger.parse(text)
        for parsed_line in parsed.split("\n"):
            if "\t" not in parsed_line:
                continue
            word, features = parsed_line.split("\t", 1)
            pos = features.split(",")[0]
            if pos in ("NNG", "NNP") and len(word) >= 2:
                noun_counter[word] += 1
        mixed_counter.update(
            t for t in mixed_pattern.findall(text) if len(t) >= 2
        )

    with out_path.open("w", encoding="utf-8") as out:
        # 1) 고정 시드 용어 (항상 포함)
        seed_words = {term for term, _ in SEED_TERMS}
        for term, pos in SEED_TERMS:
            out.write(f"{term}\t{pos}\n")

        # 2) 코퍼스 추출 명사 (시드와 중복 제외, 상위 top_n개)
        written = 0
        for term, _ in noun_counter.most_common():
            if written >= args.top_n:
                break
            if term not in seed_words:
                out.write(f"{term}\tNNG\n")
                written += 1

        # 3) 영문 혼용어 (3회 이상)
        sl_written = 0
        for term, cnt in mixed_counter.most_common():
            if cnt < 3:
                break
            if term not in seed_words:
                out.write(f"{term}\tSL\n")
                sl_written += 1

    total_seed = len(SEED_TERMS)
    print(f"시드 용어 {total_seed}개 + 코퍼스 NNG {written}개 + 영문 SL {sl_written}개 = 총 {total_seed + written + sl_written}개 → {out_path}")


if __name__ == "__main__":
    main()
