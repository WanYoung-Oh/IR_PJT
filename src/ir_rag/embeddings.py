from __future__ import annotations

from typing import Any


def embedding_model_kwargs() -> dict[str, Any]:
    import torch

    dtype = torch.float16
    try:
        import flash_attn  # noqa: F401

        return {"attn_implementation": "flash_attention_2", "torch_dtype": dtype}
    except ImportError:
        return {"attn_implementation": "sdpa", "torch_dtype": dtype}


def build_huggingface_embedding(embedding_cfg: dict) -> Any:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    mkw = embedding_model_kwargs()
    # trust_remote_code: config에서 명시적으로 설정. 기본 False (임의 코드 실행 방지).
    # HuggingFace 모델 중 일부(e.g. Qwen)는 True 필요 — config YAML에서 활성화.
    mkw["trust_remote_code"] = bool(embedding_cfg.get("trust_remote_code", False))
    return HuggingFaceEmbedding(
        model_name=embedding_cfg["model_name"],
        query_instruction=embedding_cfg.get("query_instruction", ""),
        text_instruction=embedding_cfg.get("text_instruction", ""),
        max_length=int(embedding_cfg.get("max_length", 8192)),
        model_kwargs=mkw,
    )
