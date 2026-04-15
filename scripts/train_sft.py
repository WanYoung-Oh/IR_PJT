#!/usr/bin/env python3
"""Unsloth SFT 학습 스크립트 (Qwen3.5-4B QLoRA / bf16 LoRA).

설계 문서 §⑤ LLM 응답 생성 「Unsloth SFT」 참조.

사용 예시:
    # QLoRA (빠른 실험, VRAM ~12GB)
    python scripts/train_sft.py --data artifacts/sft_data.jsonl

    # bf16 LoRA (최종 제출, VRAM ~22GB)
    python scripts/train_sft.py --data artifacts/sft_data.jsonl --no-qlora
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# unsloth는 반드시 trl/transformers/peft 보다 먼저 임포트해야 최적화가 적용됨
try:
    import unsloth  # noqa: F401
except ImportError:
    raise SystemExit(
        "학습 의존성 미설치: unsloth\n"
        "pip install -r requirements-train.txt 후 재실행하세요.\n"
        "torch는 CUDA 버전에 맞게 선설치 필요 (requirements-train.txt 상단 주석 참고)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="artifacts/sft_data.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output-dir", default="artifacts/qwen35-4b-science-rag")
    parser.add_argument("--no-qlora", action="store_true", help="bf16 LoRA 사용 (VRAM ~22GB)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise SystemExit(
            f"학습 의존성 미설치: {e}\n"
            "pip install -r requirements-train.txt 후 재실행하세요.\n"
            "torch는 CUDA 버전에 맞게 선설치 필요 (requirements-train.txt 상단 주석 참고)."
        )

    use_qlora = not args.no_qlora
    print(f"모드: {'QLoRA 4-bit' if use_qlora else 'bf16 LoRA'}")
    print(f"모델: {args.model}")

    # 1. 모델 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=use_qlora,
    )

    # 2. LoRA 어댑터 적용
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 3. 데이터셋 로드
    from ir_rag.io_util import iter_jsonl

    data_path = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    records = list(iter_jsonl(data_path))
    if not records:
        raise SystemExit(f"SFT 데이터가 비어 있습니다: {data_path}")

    # messages → Unsloth 포맷 (ChatML)
    texts = [
        tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        for r in records
    ]
    dataset = Dataset.from_dict({"text": texts})

    # 4. 학습
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            bf16=True,
            output_dir=str(output_dir),
            optim="adamw_8bit",
            warmup_steps=5,
            lr_scheduler_type="cosine",
            save_steps=100,
            logging_steps=10,
        ),
    )
    trainer.train()

    # 5. 어댑터 저장 + 병합 가중치 저장 (추론 시 PEFT 키 불일치 경고 없이 로드 가능)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"어댑터 저장 → {output_dir}")

    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"병합 가중치 저장 → {merged_dir}")
        print(
            "  (export_submission._load_llm 이 경로를 우선 로드합니다. "
            "config 의 llm.checkpoint 는 기존처럼 상위 폴더만 지정하면 됩니다.)"
        )
    except Exception as e:
        print(f"[경고] merge_and_unload 또는 병합 저장 실패 — PEFT 어댑터만 사용 가능: {e}")

    print(f"학습 완료 → {output_dir}")


if __name__ == "__main__":
    main()
