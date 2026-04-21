"""Shared Elasticsearch index utilities."""
from __future__ import annotations

from pathlib import Path

from elasticsearch import Elasticsearch

# ---------------------------------------------------------------------------
# 한국어 동의어 / 이표기 규칙
# 형식: "term1, term2"  (양방향 동등)  /  "term1 => term2"  (단방향)
# 검색 시(search_analyzer)에만 적용되므로 인덱스 팽창 없음.
# ---------------------------------------------------------------------------
KOREAN_SYNONYMS: list[str] = [
    # 국가명 이표기
    "이태리, 이탈리아",
    "미국, 미합중국",
    "영국, 잉글랜드",
    "독일, 도이칠란트",
    "프랑스, 불란서",
    "중국, 중화인민공화국",
    # IT / 코딩 복합어 — 띄어쓰기 변이
    "예외처리, 예외 처리",
    "객체지향, 객체 지향",
    "인공지능, 인공 지능",
    "기계학습, 기계 학습",
    "자연어처리, 자연어 처리",
    "딥러닝, 딥 러닝",
    "머신러닝, 머신 러닝",
    # 과학 / 의학 복합어 — 띄어쓰기 변이
    "온실효과, 온실 효과",
    "이산화탄소, 이산화 탄소",
    "줄기세포, 줄기 세포",
    "유전자편집, 유전자 편집",
    "광합성작용, 광합성 작용",
    "지구온난화, 지구 온난화",
    "블랙홀, 블랙 홀",
    # 기상
    "스톰체이서, 폭풍추적자, 폭풍 추적자",
]

# ---------------------------------------------------------------------------
# ES 인덱스 설정 (기본값; user_dictionary 는 _build_settings() 에서 주입)
# ---------------------------------------------------------------------------
ES_INDEX_SETTINGS: dict = {
    "analysis": {
        "char_filter": {
            # 연속 공백을 단일 공백으로 정규화 (양쪽 모두 적용)
            "space_normalizer": {
                "type": "pattern_replace",
                "pattern": " {2,}",
                "replacement": " ",
            }
        },
        "filter": {
            # 검색 시 동의어 확장 (인덱스 시에는 사용 안 함)
            "korean_synonyms": {
                "type": "synonym",
                "synonyms": KOREAN_SYNONYMS,
                "updateable": True,  # ES 8.x: 색인 재생성 없이 갱신 가능
            }
        },
        "analyzer": {
            # 색인용: Nori + 품사 필터 + 소문자 + 공백 정규화
            "korean_analyzer": {
                "type": "custom",
                "char_filter": ["space_normalizer"],
                "tokenizer": "nori_tok",
                "filter": ["nori_part_of_speech", "lowercase"],
            },
            # 검색용: 색인 분석기 + 동의어 확장
            "korean_search_analyzer": {
                "type": "custom",
                "char_filter": ["space_normalizer"],
                "tokenizer": "nori_tok",
                "filter": ["nori_part_of_speech", "lowercase", "korean_synonyms"],
            },
        },
        "tokenizer": {
            "nori_tok": {
                "type": "nori_tokenizer",
                "decompound_mode": "mixed",
                # user_dictionary / user_dictionary_rules 는 _build_settings() 에서 주입됨
            }
        },
    }
}

ES_INDEX_MAPPINGS: dict = {
    "properties": {
        "docid": {"type": "keyword"},
        "src": {"type": "keyword"},
        "content": {
            "type": "text",
            "analyzer": "korean_analyzer",
            "search_analyzer": "korean_search_analyzer",
        },
    }
}

# 메타 필드 추가 시 사용하는 확장 매핑 (title·keywords·summary)
ES_META_MAPPINGS: dict = {
    "properties": {
        "docid": {"type": "keyword"},
        "src": {"type": "keyword"},
        "content": {
            "type": "text",
            "analyzer": "korean_analyzer",
            "search_analyzer": "korean_search_analyzer",
        },
        "title": {
            "type": "text",
            "analyzer": "korean_analyzer",
            "search_analyzer": "korean_search_analyzer",
            # boost는 query-time에 적용 (mapping-level boost는 ES 8.x 에서 deprecated)
        },
        "keywords": {
            "type": "text",
            "analyzer": "korean_analyzer",
            "search_analyzer": "korean_search_analyzer",
        },
        "summary": {
            "type": "text",
            "analyzer": "korean_analyzer",
            "search_analyzer": "korean_search_analyzer",
        },
        "category": {"type": "keyword"},
    }
}


def _build_settings(
    user_dict_path: str | None = None,
    user_dict_rules: list[str] | None = None,
    extra_synonyms: list[str] | None = None,
    use_lm_jelinek_mercer: bool = False,
) -> dict:
    """``ES_INDEX_SETTINGS`` 기반으로 Nori 사용자 사전 설정을 주입한 설정 딕셔너리를 반환한다.

    Parameters
    ----------
    user_dict_path:
        ES 노드의 ``config/`` 디렉터리 기준 상대 경로 (예: ``"user_dict.txt"``).
        ``user_dict_rules`` 와 동시에 지정하면 ``user_dict_path`` 가 우선한다.
    user_dict_rules:
        파일 없이 용어를 인라인으로 등록할 때 사용하는 규칙 목록
        (예: ``["양자역학 NNG", "파동함수 NNG"]``).
    extra_synonyms:
        ``KOREAN_SYNONYMS`` 에 추가할 동의어 규칙 목록.
        ``artifacts/science_synonyms.txt`` 내용을 전달한다.
    use_lm_jelinek_mercer:
        ``True`` 이면 LMJelinekMercer(λ=0.7) 유사도를 적용한다.
        기본 BM25보다 문서 길이 편향이 적어 과학 도메인 긴 텍스트에 유리하다.

    Returns
    -------
    dict
        ES ``indices.create`` 에 전달할 ``settings`` 딕셔너리.
    """
    import copy
    settings = copy.deepcopy(ES_INDEX_SETTINGS)

    # Nori 사용자 사전 주입
    nori_tok = settings["analysis"]["tokenizer"]["nori_tok"]
    if user_dict_path is not None:
        nori_tok["user_dictionary"] = user_dict_path
    elif user_dict_rules is not None:
        nori_tok["user_dictionary_rules"] = user_dict_rules

    # 과학 도메인 동의어 추가
    if extra_synonyms:
        combined = list(KOREAN_SYNONYMS) + list(extra_synonyms)
        settings["analysis"]["filter"]["korean_synonyms"]["synonyms"] = combined

    # LMJelinekMercer 유사도 설정
    if use_lm_jelinek_mercer:
        settings["similarity"] = {
            "lm_jelinek_mercer": {
                "type": "LMJelinekMercer",
                "lambda": 0.69,
            }
        }

    return settings


def load_synonyms_file(path: str | Path) -> list[str]:
    """ES synonym 형식 파일을 읽어 규칙 목록을 반환한다.

    Parameters
    ----------
    path:
        ``artifacts/science_synonyms.txt`` 경로.
        빈 줄과 ``#`` 주석 줄은 무시한다.
    """
    p = Path(path)
    if not p.exists():
        return []
    rules = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rules.append(line)
    return rules


def ensure_index(
    es: Elasticsearch,
    index: str,
    *,
    recreate: bool = False,
    user_dict_path: str | None = None,
    user_dict_rules: list[str] | None = None,
    extra_synonyms: list[str] | None = None,
    use_lm_jelinek_mercer: bool = False,
    use_meta_fields: bool = False,
) -> None:
    """인덱스가 없으면 생성. recreate=True 이면 기존 인덱스를 삭제 후 재생성.

    Parameters
    ----------
    es:
        Elasticsearch 클라이언트.
    index:
        생성 또는 확인할 인덱스 이름.
    recreate:
        ``True`` 이면 인덱스가 이미 존재할 때 삭제 후 재생성한다.
    user_dict_path:
        Nori tokenizer ``user_dictionary`` 파일 경로 (ES config 기준 상대 경로).
    user_dict_rules:
        Nori tokenizer ``user_dictionary_rules`` 인라인 목록.
        ``user_dict_path`` 가 지정된 경우 무시된다.
    extra_synonyms:
        추가 동의어 규칙 목록 (``load_synonyms_file()`` 결과를 전달).
    use_lm_jelinek_mercer:
        ``True`` 이면 LMJelinekMercer 유사도를 적용한다.
    use_meta_fields:
        ``True`` 이면 title / keywords / summary / category 필드를 매핑에 추가한다.
    """
    if es.indices.exists(index=index):
        if not recreate:
            return
        es.indices.delete(index=index)

    settings = _build_settings(
        user_dict_path=user_dict_path,
        user_dict_rules=user_dict_rules,
        extra_synonyms=extra_synonyms,
        use_lm_jelinek_mercer=use_lm_jelinek_mercer,
    )

    # LMJelinekMercer를 사용할 경우 매핑에 similarity 지정
    if use_lm_jelinek_mercer:
        import copy
        mappings = copy.deepcopy(ES_META_MAPPINGS if use_meta_fields else ES_INDEX_MAPPINGS)
        for field_cfg in mappings["properties"].values():
            if field_cfg.get("type") == "text":
                field_cfg["similarity"] = "lm_jelinek_mercer"
    else:
        mappings = ES_META_MAPPINGS if use_meta_fields else ES_INDEX_MAPPINGS

    es.indices.create(index=index, settings=settings, mappings=mappings)
