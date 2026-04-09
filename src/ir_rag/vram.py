"""VRAM 관리 유틸 — 순차 언로드 전략.

설계 문서 「24GB VRAM 관리 — 순차 언로드 전략」 참조.
Embedding(~18GB) → Reranker(~18GB) → Generator(~6GB) 를 순서대로 실행하되
각 단계 후 언로드하여 VRAM 초과를 방지한다.
"""
from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def unload_model(model: object) -> None:
    """모델을 메모리에서 해제하고 CUDA 캐시를 비운다.

    Parameters
    ----------
    model:
        언로드할 PyTorch 모델 또는 LlamaIndex embed_model.
        ``None`` 을 넘겨도 안전하게 처리된다.

    Returns
    -------
    None
        반드시 반환값으로 변수를 재할당해야 호출자 참조가 끊겨 GC가 동작한다::

            reranker = unload_model(reranker)   # ← 재할당 필수
    """
    if model is None:
        return None
    try:
        import torch

        def _move_to_cpu(obj: object) -> None:
            """객체와 내부 중첩 모델을 CPU로 이동한다."""
            if obj is None:
                return
            # LlamaIndex HuggingFaceEmbedding → ._model (SentenceTransformer)
            inner = getattr(obj, "_model", None)
            if inner is not None:
                _move_to_cpu(inner)
            # SentenceTransformer → ._modules 내 실제 torch 모델
            for attr in ("_first_module", "_last_module"):
                sub = getattr(obj, attr, None)
                if sub is not None and hasattr(sub, "cpu"):
                    try:
                        sub.cpu()
                    except Exception:
                        pass
            if hasattr(obj, "cpu"):
                try:
                    obj.cpu()
                except Exception:
                    pass

        _move_to_cpu(model)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            logger.info("VRAM 해제 완료: %.1fGB 사용 중", allocated_gb)
        else:
            logger.info("VRAM 해제 완료 (CUDA 없음)")
    except ImportError:
        del model
        gc.collect()
        logger.info("모델 해제 완료 (torch 없음)")
    except Exception as e:
        logger.warning("모델 해제 중 오류 (무시): %s", e)
    return None
