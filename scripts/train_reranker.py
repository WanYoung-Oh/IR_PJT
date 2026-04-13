#!/usr/bin/env python3
"""B-3b: Reranker Fine-tuning (Qwen3-Reranker-8B QLoRA).

reranker_triplets.jsonl의 (query, positive, negatives) 트리플렛으로
Qwen3-Reranker-8B를 Binary Cross-Entropy 손실로 파인튜닝한다.

모델 구조: AutoModelForSequenceClassification (logit 1개)
손실 함수: BCEWithLogitsLoss — positive 쌍=1, negative 쌍=0
학습 방식: QLoRA 4-bit (24GB VRAM 내 가능)

사용 예시:
    python scripts/train_reranker.py \\
      --config config/default.yaml \\
      --data artifacts/reranker_triplets.jsonl \\
      --output-dir artifacts/qwen3-reranker-8b-science

    # bf16 LoRA (VRAM ~40GB)
    python scripts/train_reranker.py \\
      --data artifacts/reranker_triplets.jsonl \\
      --no-qlora
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ir_rag.config import load_config, repo_root_from, resolve_config_path
from ir_rag.io_util import iter_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_triplets(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def _build_pairs(
    triplets: list[dict],
    doc_map: dict[str, str],
    max_length: int = 512,
) -> tuple[list[str], list[str], list[float]]:
    """트리플렛 → (queries, docs, labels) 쌍 리스트로 변환한다.

    각 트리플렛은 1개의 positive 쌍 + N개의 negative 쌍으로 확장된다.
    """
    queries, docs, labels = [], [], []
    missing = 0
    for t in triplets:
        q = t["query"]
        pos_id = t["positive"]
        neg_ids = t["negatives"]

        pos_text = doc_map.get(pos_id)
        if not pos_text:
            missing += 1
            continue

        queries.append(q)
        docs.append(pos_text[:max_length])
        labels.append(1.0)

        for neg_id in neg_ids:
            neg_text = doc_map.get(neg_id)
            if neg_text:
                queries.append(q)
                docs.append(neg_text[:max_length])
                labels.append(0.0)

    if missing:
        logger.warning("문서 매핑 누락: %d건 (doc_map에 없는 positive)", missing)

    logger.info("학습 쌍 생성: %d쌍 (positive=%d, negative=%d)",
                len(labels), labels.count(1.0), labels.count(0.0))
    return queries, docs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data", default="artifacts/reranker_triplets.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--output-dir", default="artifacts/qwen3-reranker-8b-science")
    parser.add_argument("--no-qlora", action="store_true", help="bf16 LoRA 사용 (VRAM ~40GB)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise SystemExit(
            f"의존성 미설치: {e}\n"
            "pip install -r requirements-train.txt 후 재실행하세요."
        )

    root = repo_root_from(Path.cwd())
    cfg = load_config(resolve_config_path(root, args.config))
    data_path = root / args.data
    doc_path = root / cfg["paths"]["documents"]
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    use_qlora = not args.no_qlora
    logger.info("모드: %s | 모델: %s", "QLoRA 4-bit" if use_qlora else "bf16 LoRA", args.model)

    # 문서 맵 로드
    logger.info("documents.jsonl 로드 중…")
    doc_map = {d["docid"]: d["content"] for d in iter_jsonl(doc_path)}

    # 트리플렛 로드 및 쌍 생성
    logger.info("트리플렛 로드 중: %s", data_path)
    triplets = _load_triplets(data_path)
    logger.info("트리플렛 %d건 로드 완료", len(triplets))
    queries, docs, labels = _build_pairs(triplets, doc_map, max_length=args.max_seq_len)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    if use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=1,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA 어댑터
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset 구성
    from torch.utils.data import Dataset as TorchDataset

    class RerankerDataset(TorchDataset):
        def __init__(self, queries, docs, labels, tok, max_len):
            self.pairs = list(zip(queries, docs))
            self.labels = labels
            self.tok = tok
            self.max_len = max_len

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            q, d = self.pairs[idx]
            enc = self.tok(
                q, d,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor([self.labels[idx]], dtype=torch.float),
            }

    dataset = RerankerDataset(queries, docs, labels, tokenizer, args.max_seq_len)

    # 학습 — BCEWithLogitsLoss 커스텀 손실 사용
    class RerankerTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            label = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits  # (batch, 1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, label
            )
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # 어댑터 저장
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Reranker 파인튜닝 완료 → %s", out_dir)
    logger.info(
        "적용 방법: config/default.yaml의 reranker.model_name을 '%s'로 변경",
        out_dir,
    )


if __name__ == "__main__":
    main()
