#!/usr/bin/env python3
"""vLLM OpenAI 호환 API 서버 기동 스크립트.

설계 문서 「파이프라인 오케스트레이션 & 인프라 — 모델 서빙: vLLM」 참조.
SFT로 생성된 GGUF Q4_K_M 모델을 OpenAI API 호환으로 서빙한다.

사용 예시:
    # 기본 (artifacts/qwen35-4b-science-rag 에서 GGUF 자동 탐색)
    python scripts/serve_vllm.py

    # 경로 직접 지정
    python scripts/serve_vllm.py --model artifacts/qwen35-4b-science-rag/model-Q4_K_M.gguf

클라이언트 연결:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    resp = client.chat.completions.create(model="science-rag", messages=[...])
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def find_gguf(base_dir: Path) -> Path | None:
    """디렉터리에서 Q4_K_M GGUF 파일을 탐색한다."""
    patterns = ["*Q4_K_M*.gguf", "*.gguf"]
    for pattern in patterns:
        matches = sorted(glob.glob(str(base_dir / "**" / pattern), recursive=True))
        if matches:
            return Path(matches[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="GGUF 파일 경로 또는 HuggingFace 모델 ID. "
             "미지정 시 artifacts/qwen35-4b-science-rag 에서 자동 탐색.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="science-rag")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    # vLLM 설치 여부 확인
    try:
        import vllm  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"vLLM 미설치: {e}\n"
            "pip install -r requirements-vllm.txt (별도 venv 권장) 후 재실행하세요."
        )

    # 모델 경로 결정
    model_path = args.model
    if model_path is None:
        default_dir = ROOT / "artifacts" / "qwen35-4b-science-rag"
        gguf = find_gguf(default_dir)
        if gguf:
            model_path = str(gguf)
            print(f"GGUF 파일 자동 탐색: {gguf}")
        else:
            model_path = "Qwen/Qwen3.5-4B"
            print(f"GGUF 없음 — HuggingFace 모델 사용: {model_path}")

    print(f"vLLM 서버 기동: {model_path}  ({args.host}:{args.port})")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", args.host,
        "--port", str(args.port),
        "--served-model-name", args.served_model_name,
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
