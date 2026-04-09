from __future__ import annotations

import re

# LaTeX 수식: 과학 문서 특화 전처리. 일반 달러 기호($5 등)는 이 코퍼스에 없다고 가정.
_LATEX_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_LATEX_INLINE = re.compile(r"\$(.+?)\$")

# "참고문헌" / "References" 등이 줄 시작에 단독으로 등장할 때만 이후 내용 제거.
# 본문 중간 단어(예: "이 References를 참고하면...")는 영향받지 않음.
_REFERENCES_SECTION = re.compile(
    r"^(참고문헌|References|Bibliography)\s*$.*",
    re.DOTALL | re.MULTILINE,
)


def preprocess_science_doc(text: str) -> str:
    text = _LATEX_BLOCK.sub(r"[수식: \1]", text)
    text = _LATEX_INLINE.sub(r"[수식: \1]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = _REFERENCES_SECTION.sub("", text)
    return text.strip()
