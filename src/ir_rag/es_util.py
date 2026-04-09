"""Shared Elasticsearch index utilities."""
from __future__ import annotations

from elasticsearch import Elasticsearch

ES_INDEX_SETTINGS = {
    "analysis": {
        "analyzer": {
            "korean_analyzer": {
                "type": "custom",
                "tokenizer": "nori_tok",
                "filter": ["nori_part_of_speech", "lowercase"],
            }
        },
        "tokenizer": {
            "nori_tok": {
                "type": "nori_tokenizer",
                "decompound_mode": "mixed",
                # user_dictionary / user_dictionary_rules 는 ensure_index() 에서 주입됨
            }
        },
    }
}

ES_INDEX_MAPPINGS = {
    "properties": {
        "docid": {"type": "keyword"},
        "src": {"type": "keyword"},
        "content": {"type": "text", "analyzer": "korean_analyzer"},
    }
}


def _build_settings(
    user_dict_path: str | None = None,
    user_dict_rules: list[str] | None = None,
) -> dict:
    """``ES_INDEX_SETTINGS`` 기반으로 Nori 사용자 사전 설정을 주입한 설정 딕셔너리를 반환한다.

    Parameters
    ----------
    user_dict_path:
        ES 노드의 ``config/`` 디렉터리 기준 상대 경로 (예: ``"user_dict.txt"``).
        직접 설치 환경에서는 ``/etc/elasticsearch/`` 기준 상대 경로가 된다.
        ``user_dict_rules`` 와 동시에 지정하면 ``user_dict_path`` 가 우선한다.
    user_dict_rules:
        파일 없이 용어를 인라인으로 등록할 때 사용하는 규칙 목록
        (예: ``["양자역학", "파동함수"]``).
        빠른 실험 시 편리하다.

    Returns
    -------
    dict
        ES ``indices.create`` 에 전달할 ``settings`` 딕셔너리.
    """
    import copy
    settings = copy.deepcopy(ES_INDEX_SETTINGS)
    nori_tok = settings["analysis"]["tokenizer"]["nori_tok"]
    if user_dict_path is not None:
        nori_tok["user_dictionary"] = user_dict_path
    elif user_dict_rules is not None:
        nori_tok["user_dictionary_rules"] = user_dict_rules
    return settings


def ensure_index(
    es: Elasticsearch,
    index: str,
    *,
    recreate: bool = False,
    user_dict_path: str | None = None,
    user_dict_rules: list[str] | None = None,
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
        지정하면 인덱스 생성 시 tokenizer 설정에 포함된다.
    user_dict_rules:
        Nori tokenizer ``user_dictionary_rules`` 인라인 목록.
        ``user_dict_path`` 가 지정된 경우 무시된다.
    """
    if es.indices.exists(index=index):
        if not recreate:
            return
        es.indices.delete(index=index)
    settings = _build_settings(
        user_dict_path=user_dict_path,
        user_dict_rules=user_dict_rules,
    )
    es.indices.create(index=index, settings=settings, mappings=ES_INDEX_MAPPINGS)
