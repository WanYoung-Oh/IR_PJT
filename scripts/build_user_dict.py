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
        for term, _ in noun_counter.most_common(args.top_n):
            out.write(f"{term}\tNNG\n")
        for term, cnt in mixed_counter.most_common():
            if cnt < 5:
                break
            out.write(f"{term}\tSL\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
