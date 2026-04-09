"""리더보드 제출 한 줄 스키마 (베이스라인 ``sample_submission.csv`` 와 동일 키)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SUBMISSION_REQUIRED_KEYS = frozenset(
    {"eval_id", "standalone_query", "topk", "answer", "references"}
)


@dataclass
class SubmissionRecord:
    eval_id: int
    standalone_query: str
    topk: list[str]
    answer: str
    references: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # references: list[{score, content}]
        return d


def validate_submission_row(row: dict[str, Any]) -> None:
    missing = SUBMISSION_REQUIRED_KEYS - row.keys()
    if missing:
        raise ValueError(f"제출 행에 필수 키 누락: {sorted(missing)}")
    if not isinstance(row["topk"], list):
        raise TypeError("topk must be list[str]")
    if not isinstance(row["references"], list):
        raise TypeError("references must be list")
