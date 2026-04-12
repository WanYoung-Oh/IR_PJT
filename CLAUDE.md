# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IR is a Python-based RAG (Retrieval-Augmented Generation) pipeline for a scientific Q&A competition. It processes 220 evaluation queries against 4,272 reference documents using hybrid search, reranking, and LLM generation. The target metric is a modified MAP (Mean Average Precision).

## Environment Setup

Two separate virtual environments are required due to PyTorch version conflicts:

```bash
# Primary RAG environment (indexing, retrieval, training, evaluation)
python3.10 -m venv py310 && source py310/bin/activate
pip install -e .
pip install -r requirements-train.txt

# vLLM serving environment (only if using serve_vllm.py)
python3.10 -m venv py310-vllm && source py310-vllm/bin/activate
pip install -r requirements-vllm.txt
```

System dependency: `sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8` plus mecab-ko-dic.

Copy `.env.example` to `.env` and set: `ES_PASSWORD`, `HF_TOKEN`, and `GOOGLE_API_KEY` or `OPENAI_API_KEY`.

## Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_config.py -v

# Run a single test by name
python -m pytest tests/test_config.py::test_validate_config_passes_when_all_keys_present -v

# Smoke test (ES + BM25 only, no GPU)
python scripts/smoke_e2e.py --config config/default.yaml

# Build Elasticsearch index
python scripts/index_es.py --config config/default.yaml

# Build Qdrant vector index (GPU required)
python scripts/index_qdrant.py --config config/default.yaml [--force]

# Run full pipeline (GPU required)
python scripts/export_submission.py --pipeline --config config/default.yaml

# Validate submission format
python scripts/validate_submission.py artifacts/sample_submission.csv
```

## Architecture

The pipeline runs in 6 sequential phases, designed to fit within a 24GB VRAM constraint by loading/unloading models one at a time (managed by `src/ir_rag/vram.py`):

1. **Phase 0 — Query Rewriting** (`query_rewrite.py`): Classifies each query as "science" or "chat". For science queries, rewrites multi-turn conversation to a standalone query, generates HyDE document, and produces alternative query formulations. Output cached to `artifacts/phase0_queries.csv`.

2. **Phase 1 — Hybrid Retrieval** (`retrieval.py`): BM25 sparse search via Elasticsearch (Nori + MeCab Korean tokenizer) merged with dense search via Qdrant (Qwen3-Embedding-8B) using RRF (k=20). Returns Top-20 candidate documents.

3. **Phase 2 — Reranking** (`reranker.py`): Qwen3-Reranker-8B cross-encoder scores all Top-20 candidates. Final score: `0.7 × reranker_score + 0.3 × rrf_score`.

4. **Phase 3 — Answer Generation** (`generator.py`): Science queries use CRAG prompt (relevance check) → CoT reasoning → faithfulness self-check via RAGAS (≥ 0.7 threshold). Chat queries get direct conversational responses.

5. **Phase 4 — Evaluation** (optional): Local MAP via pseudo-relevance (`run_retrieval_eval.py`), official MAP with ground truth (`run_competition_map.py`), RAGAS quality metrics (`ragas_eval.py`).

6. **Phase 5 — Submission** (`submission.py`, `export_submission.py`): CSV output with fields `eval_id`, `standalone_query`, `topk` (3 docs), `answer`, `references`.

## Key Modules (`src/ir_rag/`)

| Module                                   | Purpose                                                   |
| ---------------------------------------- | --------------------------------------------------------- |
| `config.py`                              | YAML loading, path resolution, required key validation    |
| `io_util.py`                             | `iter_jsonl` / `write_jsonl` helpers                      |
| `preprocess.py`                          | LaTeX/reference removal from documents                    |
| `query_rewrite.py`                       | Science classification, HyDE, standalone query generation |
| `retrieval.py`                           | BM25, dense search, RRF fusion                            |
| `reranker.py`                            | Cross-encoder scoring                                     |
| `generator.py`                           | CRAG/CoT prompts and LLM inference                        |
| `vram.py`                                | `unload_model()` for sequential model memory management   |
| `eval_map.py` / `competition_metrics.py` | Local and official MAP computation                        |

## Patterns

All scripts follow this entry point pattern:

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
```

Config is always loaded as: `cfg = load_config(resolve_config_path(root, args.config))`. All settings (ES/Qdrant URLs, model names, data paths) live in `config/default.yaml`.

JSONL I/O: `from ir_rag.io_util import iter_jsonl, write_jsonl`.

## Tech Stack

- **LLM**: Qwen3.5-4B (HuggingFace) for query rewriting and answer generation
- **Embeddings**: Qwen3-Embedding-8B via Qdrant
- **Reranker**: Qwen3-Reranker-8B (cross-encoder)
- **Sparse search**: Elasticsearch 8.19 with Nori (Korean) analyzer
- **Vector DB**: Qdrant 1.17
- **Training**: Unsloth + PEFT (LoRA) for SFT fine-tuning
- **Evaluation**: RAGAS 0.4.3 (Faithfulness, Relevancy, Recall)
- **Korean NLP**: MeCab + mecab-ko-dic
- **Framework**: LlamaIndex 0.13+

## Data (git-ignored)

- `data/documents.jsonl` — 4,272 reference science documents
- `data/eval.jsonl` — 220 evaluation queries
- `artifacts/` — generated outputs (phase caches, submission CSV, fine-tuned model)
