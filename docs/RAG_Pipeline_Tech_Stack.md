# 과학 지식 기반 RAG 파이프라인 — 기술 스택

> 목적: 과학 지식 기반 질의응답 경진대회 (MAP 평가지표 기반)
> 실행 환경: Ubuntu / **Python 3.10.x** / 단일 GPU 24GB VRAM (가상환경 권장: `python3.10 -m venv .venv`)
> 라이선스: Apache 2.0

**의존성·런타임 매트릭스 (요약)**

| 용도 | 가상환경 | PyTorch | 설치 파일 |
|------|----------|---------|-----------|
| RAG 코어·SFT 학습 | `.venv` (기본) | CUDA용 **torch 2.3.1** ([requirements-train.txt](../requirements-train.txt) 상단 주석 참고) | `pip install -r requirements-train.txt` |
| vLLM OpenAI 호환 서빙 | `.venv-vllm` (**별도 필수**) | vLLM이 끌어오는 **torch 2.10.x** | `pip install -r requirements-vllm.txt` |

같은 venv에 **torch 2.3.1(학습/임베딩)** 과 **vLLM 0.19.x(동봉 torch 2.10.x)** 를 동시에 맞추기 어렵습니다. 서빙만 별도 venv로 두세요.

---

## 데이터셋 명세

### eval.jsonl — 평가 쿼리

| 항목 | 내용 |
|---|---|
| 총 샘플 수 | 220건 |
| 단일 턴 | 200건 (91%) |
| 멀티 턴 (최대 3턴) | 20건 (9%) |
| **치챗 (일상 대화)** | **약 20건** — 과학과 무관, 문서 검색 불필요 |
| eval_id 범위 | 0 ~ 309 |
| 필드 | `eval_id`, `msg` (role/content 리스트) |

```jsonc
// 단일 턴 — 과학
{"eval_id": 78, "msg": [{"role": "user", "content": "나무의 분류에 대해 조사해 보기 위한 방법은?"}]}

// 멀티 턴 — 과학
{"eval_id": 107, "msg": [
  {"role": "user",      "content": "기억 상실증 걸리면 너무 무섭겠다."},
  {"role": "assistant", "content": "네 맞습니다."},
  {"role": "user",      "content": "어떤 원인 때문에 발생하는지 궁금해."}
]}

// 치챗 — 일상 대화 (검색·리랭킹 불필요)
{"eval_id": 42, "msg": [{"role": "user", "content": "오늘 날씨가 너무 좋네요!"}]}
```

### documents.jsonl — 참조 문서 코퍼스

| 항목 | 내용 |
|---|---|
| 총 문서 수 | 4,272건 |
| 필드 | `docid`, `src`, `content` |
| content 평균 길이 | 315자 |
| content 길이 분포 | 200~500자 구간 집중 (89%) |
| 소스 종류 | 63종 |

**소스 구성 (상위)**

| 소스 계열 | 문서 수 | 도메인 |
|---|---|---|
| ko_ai2_arc__ARC_Challenge (test/train/val) | 2,047건 (48%) | 초중고 과학 |
| ko_mmlu__conceptual_physics | 211건 | 개념 물리 |
| ko_mmlu__nutrition | 168건 | 영양학 |
| ko_mmlu__human_aging | 168건 | 노화 의학 |
| ko_mmlu__high_school_biology | 131건 | 고교 생물 |
| ko_mmlu__astronomy | 122건 | 천문학 |
| ko_mmlu__high_school_chemistry | 118건 | 고교 화학 |
| 기타 ko_mmlu 계열 | ~1,107건 | 의학·컴퓨터·전기 등 |

### 데이터셋 기반 설계 결정사항

1. **청킹 불필요** — content 평균 315자, `docid` 단위 직접 인덱싱.
2. **멀티 턴 쿼리 처리 필수** — 20건(9%)은 마지막 발화만으로 검색 시 컨텍스트 유실.
3. **치챗 분류 필수** — 약 20건이 일상 대화. 검색·리랭킹을 강제 실행하면 topk에 무관련 문서가 채워져 MAP 손실. Phase 0에서 `is_science_question()`으로 사전 분류.
4. **도메인 집중** — ko_mmlu + ko_ai2_arc 한국어 과학. MeCab + 전문 용어 사전에 직접 활용.

---

## Ubuntu 환경 초기 설정

기준 인터프리터는 **Python 3.10** 입니다. 3.11+는 일부 휠/바이너리 조합에서 검증되지 않을 수 있습니다.

```bash
# Python 3.10 (Ubuntu 패키지 예시)
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

# MeCab 설치
sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8

# mecab-ko-dic 설치
wget -O mecab-ko-dic.tar.gz \
  "https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz"
tar -xzf mecab-ko-dic.tar.gz
cd mecab-ko-dic-2.1.1-20180720 && ./configure && make && sudo make install && cd ..

pip install mecab-python3==1.0.12

# 설치 확인
echo "양자역학의 파동함수" | mecab

# Elasticsearch Nori 플러그인 (ES 서버에서 실행)
# bin/elasticsearch-plugin install analysis-nori
```

---

## 전체 파이프라인 플로우

```
eval.jsonl (220건: 단일 200 / 멀티턴 20)
    │
    ▼
① 쿼리 전처리
    단일 턴: 마지막 user 발화 그대로
    멀티 턴: Few-shot LLM 재작성 → 독립 쿼리
    │
    ▼
② 문서 검색 (Hybrid Retrieval)
    Sparse: Elasticsearch BM25 + Nori(mixed) + MeCab 전문 용어 사전
    Dense:  Qwen3-Embedding-8B + Qdrant (HNSW)
    원본 쿼리 + HyDE → RRF(k=60) 3축 병합 → Top-100
    │
    ▼
③ 재순위화 (Reranking)
    Qwen3-Reranker-8B Cross-Encoder
    + Soft Voting (Reranker 0.7 : RRF 0.3)
    → Top-10~20
    │
    ▼
④ LLM 응답 생성
    Qwen3.5-4B (SFT: QLoRA → 최종 bf16 LoRA) 또는 Qwen3.5-Plus API
    CRAG / CoT 프롬프트
    │
    ▼
⑤ 평가
    MAP · MRR · NDCG · Faithfulness · Answer Relevancy
    evaluate_map(eval.jsonl) + RAGAS + LangSmith

참조: documents.jsonl (4,272건 / docid 단위 직접 인덱싱)
```

---

## 단계별 기술 스택

### ① 쿼리 전처리 (Query Preprocessing)

멀티 턴 쿼리(20건)를 검색에 적합한 단일 쿼리로 변환합니다.
Few-shot 예시로 LLM이 과학 도메인 의도를 과도하게 확장하지 않도록 제어합니다.

```python
FEW_SHOT_EXAMPLES = """예시 1)
대화:
[user] 광합성은 어디서 일어나?
[assistant] 엽록체에서 일어납니다.
[user] 그 과정을 더 자세히 설명해줘.
검색 쿼리: 광합성 과정 엽록체 명반응 암반응

예시 2)
대화:
[user] 뉴턴의 법칙에는 뭐가 있어?
[assistant] 운동 제1, 2, 3 법칙이 있습니다.
[user] 제2 법칙이 뭔지 알려줘.
검색 쿼리: 뉴턴 운동 제2 법칙 F=ma 가속도"""


def _is_valid_query(query: str, original: str, min_len: int = 4) -> bool:
    if len(query) < min_len:
        return False
    if len(query) > len(original) * 3:
        return False
    stopwords = {"네", "아", "음", "그렇군요", "맞습니다", "안녕"}
    if query.strip() in stopwords:
        return False
    return True


def build_search_query(msg: list[dict], llm=None) -> str:
    user_msgs = [m for m in msg if m["role"] == "user"]
    original  = user_msgs[-1]["content"]

    if len(user_msgs) == 1:
        return original

    if llm:
        dialogue = "\n".join(f'[{m["role"]}] {m["content"]}' for m in msg)
        prompt = f"""{FEW_SHOT_EXAMPLES}

다음 대화에서 마지막 질문의 의도를 파악해 독립적인 검색 쿼리를 한 줄로만 생성하세요.
쿼리는 핵심 과학 용어 위주로 간결하게 작성하고, 불필요한 감정 표현은 제거하세요.

대화:
{dialogue}
검색 쿼리:"""
        rewritten = llm.complete(prompt).text.strip().split("\n")[0]

        if _is_valid_query(rewritten, original):
            return rewritten
        else:
            print(f"[fallback] 재작성 실패, 원본 사용: '{original}' (재작성: '{rewritten}')")
            return original

    # fallback: 단순 concat
    context = " ".join(m["content"] for m in msg[:-1])
    return f"{context} {original}"
```

**HyDE — BM25 스코어 기반 조건부 실행**

BM25 최고 스코어가 임계값 이상이면 HyDE를 건너뜁니다.

```python
def get_bm25_top_score(query: str, es_client, index: str = "science_docs") -> float:
    # elasticsearch-py 8.x: query / size 는 키워드 인자 사용
    resp = es_client.search(
        index=index,
        query={"match": {"content": query}},
        size=1,
        source=False,
    )
    hits = resp["hits"]["hits"]
    return hits[0]["_score"] if hits else 0.0


def hybrid_search_with_hyde(query: str, llm, retriever_fn,
                             es_client, hyde_threshold: float = 5.0) -> dict[str, float]:
    """
    hyde_threshold 튜닝 기준:
      높게(10+): 거의 항상 HyDE 실행 / 낮게(2~3): 검색 미흡 시만 실행 / 기본값 5.0: 균형
    """
    results_original = retriever_fn(query)
    bm25_score       = get_bm25_top_score(query, es_client)

    if bm25_score >= hyde_threshold:
        return rrf_score([results_original])

    hyde_prompt = (
        f"다음 과학 질문에 대한 정확한 설명 문단을 100자 내외로 작성하세요.\n"
        f"질문: {query}\n설명:"
    )
    hyde_doc     = llm.complete(hyde_prompt).text.strip()
    results_hyde = retriever_fn(hyde_doc)

    fusion_prompt = (
        f"다음 과학 질문을 다른 표현으로 재작성하세요 (1개만).\n"
        f"질문: {query}\n재작성:"
    )
    alt_query   = llm.complete(fusion_prompt).text.strip()
    results_alt = retriever_fn(alt_query)

    return rrf_score([results_original, results_hyde, results_alt])


def rrf_score(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
```

---

### ② 문서 검색 (Document Retrieval)

| 항목 | 선택 기술 | 세부 내용 |
|---|---|---|
| 희소 검색 | Elasticsearch / BM25 | 전문 용어·정확 매칭 |
| 한국어 형태소 분석 | Elasticsearch Nori + MeCab | Ubuntu 환경 권장 |
| 밀집 검색 | Qdrant + ANN (HNSW) | 의미 기반 검색 |
| 결과 병합 | RRF (k=60) | Sparse + Dense + HyDE 3축 통합 |

#### 인덱싱 — docid 단위 직접 색인

```python
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch("http://localhost:9200")

# Nori 사용자 사전:
#  - 파일 방식: ES 노드의 config 경로(예: config/user_dict.txt)에 두고
#    tokenizer 설정에서 "user_dictionary": "user_dict.txt" (상대 경로는 config 기준)
#  - 직접 설치 시 /etc/elasticsearch/ 하위가 config 루트 (경로 상대 기준)
#  - 빠른 실험: "user_dictionary_rules" 에 용어를 한 줄씩 나열 (아래 주석 예시)
index_settings = {
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
                # "user_dictionary": "user_dict.txt",
                # "user_dictionary_rules": ["양자역학", "파동함수"],
            }
        },
    }
}
mappings = {
    "properties": {
        "docid":   {"type": "keyword"},
        "src":     {"type": "keyword"},
        "content": {"type": "text", "analyzer": "korean_analyzer"},
    }
}

# elasticsearch-py 8.x: indices.create(settings=..., mappings=..., ignore_unavailable 등)
if not es.indices.exists(index="science_docs"):
    es.indices.create(index="science_docs", settings=index_settings, mappings=mappings)

def gen_actions(filepath: str):
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            if not doc:
                continue
            yield {
                "_index": "science_docs",
                "_id": doc["docid"],
                "_source": {
                    "docid": doc["docid"],
                    "src": doc["src"],
                    "content": doc["content"],
                },
            }

bulk(es, gen_actions("documents.jsonl"))
es.indices.refresh(index="science_docs")
```

#### 한국어 형태소 분석 — Nori + MeCab

| 검색 방식 | Nori 필요 여부 |
|---|---|
| Elasticsearch BM25 | **필수** — 기본 토크나이저는 한글을 음절 단위로 쪼갬 |
| Qdrant Dense 검색 | 불필요 — Qwen3-Embedding이 내부적으로 처리 |

```
❌ standard 토크나이저:
"양", "자", "역", "학", "의", "파", "동", "함", "수", "란"

✅ Nori(mixed) + 전문 용어 사전:
"양자역학", "양자", "역학", "파동함수", "파동", "함수"
```

#### 과학 전문 용어 사전 (user_dict.txt)

```
# 형식: 단어<TAB>품사  (NNG = 일반명사, SL = 외래어)

광합성     NNG
산화환원   NNG
진화론     NNG
유전자     NNG
세포분열   NNG
전기회로   NNG
중력가속도 NNG
파동함수       NNG
양자얽힘       NNG
사건의지평선   NNG
슈뢰딩거방정식 NNG
힉스보존       NNG
에너지균형     NNG
암흑에너지     NNG
블랙홀         NNG
빅뱅           NNG
CRISPR  SL
mRNA    SL
DNA     SL
RNA     SL
ATP     SL
pH      SL
```

#### 전문 용어 사전 자동 추출 — MeCab + 영문 혼용어

```python
import json
import re
from collections import Counter
import MeCab

def build_user_dict_from_corpus(doc_path: str, out_path: str = "user_dict.txt",
                                 top_n: int = 500) -> None:
    tagger = MeCab.Tagger()
    noun_counter  = Counter()
    mixed_counter = Counter()
    mixed_pattern = re.compile(r'\b[A-Za-z][A-Za-z0-9\-]{1,}\b')

    with open(doc_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            if not doc:
                continue
            text = doc["content"]

            parsed = tagger.parse(text)
            for parsed_line in parsed.split("\n"):
                if "\t" not in parsed_line:
                    continue
                word, features = parsed_line.split("\t", 1)
                pos = features.split(",")[0]
                if pos in ("NNG", "NNP") and len(word) >= 2:
                    noun_counter[word] += 1

            mixed_counter.update(
                t for t in mixed_pattern.findall(text) if len(t) >= 2
            )

    with open(out_path, "w", encoding="utf-8") as f:
        for term, _ in noun_counter.most_common(top_n):
            f.write(f"{term}\tNNG\n")
        for term, cnt in mixed_counter.most_common():
            if cnt < 5:
                break
            f.write(f"{term}\tSL\n")

    print(f"사전 생성 완료 → {out_path}")

build_user_dict_from_corpus("documents.jsonl")
```

> 추출 후 오류 용어 수동 검토를 권장합니다.

---

### ③ 임베딩 / 벡터 검색 (Embedding & Vector Search)

| 항목 | 선택 기술 | 세부 내용 |
|---|---|---|
| 임베딩 모델 | **Qwen3-Embedding-8B** | MTEB 다국어 리더보드 1위 (70.58점) |
| 벡터 DB | **Qdrant** | LlamaIndex 네이티브, 하이브리드 검색 내장 |
| 임베딩 차원 | 4096 (MRL로 축소 가능) | 4096 → 512 등 자유롭게 축소 가능 |
| 컨텍스트 길이 | 최대 32K tokens | 평균 315자 문서는 여유 있게 처리 |
| VRAM | ~18GB (FP16) | 24GB GPU에서 안정적 실행 |

**임베딩 로더 통일 원칙**

- **색인·Pseudo 라벨·오프라인 유사도**는 가능한 한 **동일 모듈**(예: 레포의 `ir_rag.embeddings`)에서 로드합니다.
- `SentenceTransformer("Qwen/...")` 만 단독으로 쓰면 LlamaIndex 경로와 **출력 정규화·프롬프트**가 어긋날 수 있습니다.

**LlamaIndex 연동 + Qdrant 인덱싱**

```python
import json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def _embedding_model_kwargs():
    try:
        import flash_attn  # noqa: F401 — wheel 설치 시에만
        return {"attn_implementation": "flash_attention_2"}
    except ImportError:
        # torch 2.x 기본: sdpa; 문제 시 "eager" 로 낮추기
        return {"attn_implementation": "sdpa"}

Settings.embed_model = HuggingFaceEmbedding(
    model_name="Qwen/Qwen3-Embedding-8B",
    query_instruction=(
        "Instruct: Given a scientific question, retrieve relevant passages "
        "that answer the query\nQuery: "
    ),
    text_instruction="Represent this scientific document for retrieval: ",
    max_length=8192,
    model_kwargs=_embedding_model_kwargs(),
    trust_remote_code=True,
)

client = QdrantClient(url="http://localhost:6333")
client.recreate_collection(
    collection_name="science_docs",
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)

def load_documents(filepath: str) -> list[Document]:
    docs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            if not d:
                continue
            docs.append(Document(
                text=d["content"],
                doc_id=d["docid"],
                metadata={"src": d["src"], "docid": d["docid"]},
            ))
    return docs

documents = load_documents("documents.jsonl")
vector_store = QdrantVectorStore(client=client, collection_name="science_docs")
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
print(f"Qdrant 색인 완료: {len(documents)}건")
```

> Instruction은 영어로 작성 권장 (모델 학습 기준)

**모델 크기별 비교**

| 모델 | 임베딩 차원 | VRAM (FP16) | 추천 용도 |
|---|---|---|---|
| Qwen3-Embedding-0.6B | 1024 | ~2GB | 빠른 프로토타입 |
| Qwen3-Embedding-4B | 2560 | ~10GB | 균형 |
| **Qwen3-Embedding-8B** | **4096** | **~18GB** | **경진대회 추천** |

---

### ④ 재순위화 (Reranking)

| 모델 | VRAM | 추천 상황 |
|---|---|---|
| Qwen3-Reranker-4B | ~10GB | 초반 빠른 실험 |
| **Qwen3-Reranker-8B** | **~18GB** | **최종 제출** |
| Cohere Rerank 3 (API) | 0GB | GPU 없이 즉시 사용 |

> 4,272건 규모에서 4B와 8B의 MAP 차이는 1% 미만인 경우가 많습니다.
> 초반에는 4B, 최종 제출 전 8B로 전환하는 전략을 권장합니다.

| 항목 | 선택 기술 | 세부 내용 |
|---|---|---|
| 로컬 리랭커 (실험) | Qwen3-Reranker-4B | VRAM ~10GB |
| 로컬 리랭커 (최종) | **Qwen3-Reranker-8B** | VRAM ~18GB |
| API 대안 | Cohere Rerank 3 | GPU 불필요 |
| 기본 전략 | 2-stage | Top-100 → Top-10~20 |
| 심화 전략 | Soft Voting | Reranker + RRF 가중 결합 |

**Soft Voting — MinMax 정규화 후 가중 결합**

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def soft_voting_rerank(rrf_scores: dict[str, float],
                        reranker_scores: dict[str, float],
                        w_reranker: float = 0.7) -> dict[str, float]:
    """w_reranker 튜닝 권장 범위: 0.6~0.8"""
    doc_ids  = list(set(rrf_scores) | set(reranker_scores))
    rrf_vals = np.array([rrf_scores.get(d, 0.0) for d in doc_ids]).reshape(-1, 1)
    rer_vals = np.array([reranker_scores.get(d, 0.0) for d in doc_ids]).reshape(-1, 1)

    rrf_norm = MinMaxScaler().fit_transform(rrf_vals).flatten()
    rer_norm = MinMaxScaler().fit_transform(rer_vals).flatten()

    combined = w_reranker * rer_norm + (1 - w_reranker) * rrf_norm
    return dict(sorted(zip(doc_ids, combined), key=lambda x: x[1], reverse=True))
```

> Reranker 단독 MAP과 반드시 비교 실험 후 도입 여부를 결정하세요.

---

### ⑤ LLM 응답 생성 (Generation)

#### 5-1. 초기 실험 — 상용 API

| 모델 | 컨텍스트 | 추천 용도 |
|---|---|---|
| **Qwen3.5-Plus** | 1M | 프롬프트 iterate 최적 |
| Claude 3.7 Sonnet | 200K | 높은 정확도 필요 시 |
| Gemini 2.5 Flash | 1M | 대용량 실험 |

#### 5-2. 최종 운영 — 로컬 SFT

| 단계 | 방법 | VRAM |
|---|---|---|
| **빠른 실험 반복** | **QLoRA 4-bit** | **~12GB** |
| **최종 제출 전** | **bf16 LoRA** | **~22GB** |

> QLoRA와 bf16 LoRA의 어댑터 정밀도는 동일합니다. 차이는 베이스 모델 정밀도이며, 최종 제출 직전에만 bf16으로 전환합니다.

| 항목 | 선택 | 세부 내용 |
|---|---|---|
| **기본 모델** | **Qwen3.5-4B** | GPQA Diamond 81.7%, MMLU-Pro 82.5% |
| SFT 프레임워크 | Unsloth | 1.5× 속도, 50% VRAM 절약 |
| 내보내기 | GGUF Q4_K_M | 추론 시 ~6GB |

**SFT 데이터 변환**

```python
import json

def build_sft_dataset(eval_path: str, doc_path: str,
                      retriever, llm=None, out_path: str = "sft_data.jsonl") -> None:
    doc_map = {}
    with open(doc_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            doc_map[d["docid"]] = d["content"]

    sft_records = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            msg    = sample["msg"]
            # 멀티 턴 검색 쿼리 전략은 학습 데이터에도 동일하게 적용 (①절과 일치)
            query  = build_search_query(msg, llm=llm)

            top_docids = retriever(query, top_k=5)
            context = "\n\n".join(
                f"[문서 {i+1}] {doc_map[did]}"
                for i, did in enumerate(top_docids) if did in doc_map
            )

            messages = [{"role": "system", "content":
                "당신은 과학 전문가입니다. 제공된 참고 문서를 근거로 질문에 답하세요. "
                "문서에 없는 내용은 절대 추측하지 마세요."}]

            for m in msg[:-1]:
                messages.append({"role": m["role"], "content": m["content"]})

            messages.append({"role": "user",
                "content": f"참고 문서:\n{context}\n\n질문: {msg[-1]['content']}"})

            sft_records.append({"messages": messages})

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"SFT 데이터 생성 완료: {len(sft_records)}건 → {out_path}")
```

**Unsloth SFT**

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# QLoRA (빠른 실험): load_in_4bit=True  → VRAM ~12GB
# bf16 LoRA (최종) : load_in_4bit=False → VRAM ~22GB
USE_QLORA = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=USE_QLORA,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=8 if USE_QLORA else 4,
        gradient_accumulation_steps=2 if USE_QLORA else 4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        output_dir="./qwen35-4b-science-rag",
        optim="adamw_8bit",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
    ),
)
trainer.train()
model.save_pretrained_gguf("qwen35-4b-science-rag", tokenizer,
                            quantization_method="q4_k_m")
```

> Qwen3.5-4B은 thinking 기본 비활성화. 필요 시 `enable_thinking=True` 명시.

**프롬프트 전략**

```python
CRAG_PROMPT = """아래 참고 문서를 검토하고, 질문에 답하기 전 각 문서의 관련성을 평가하세요.

관련성 없는 문서가 있다면 [무관련]으로 표시하고 해당 문서는 무시하세요.
모든 문서가 관련 없다면 "검색 결과 불충분"이라고 응답하세요.

참고 문서:
{context}

질문: {question}

답변 형식:
- 사용한 문서: [번호 목록]
- 답변: [답변 내용]
- 확신도: [높음/중간/낮음]"""

SCIENCE_QA_PROMPT = """당신은 과학 전문가입니다. 아래 문서를 근거로 질문에 답하세요.

규칙:
1. 반드시 제공된 문서 내용만을 근거로 답하세요
2. 추론 과정을 단계별로 서술하세요
3. 각 주장에 [문서 N] 형식으로 출처를 명시하세요
4. 문서에 없는 내용은 "문서에 근거 없음"으로 표기하세요

문서: {context}
질문: {question}"""
```

**치챗(일상 대화) 처리**

eval.jsonl의 약 20건은 과학과 무관한 일상 대화입니다. 이 경우 검색·리랭킹을 건너뛰고 자연스러운 대화 응답을 생성합니다.

```python
CHITCHAT_PROMPT = """당신은 친절한 AI 어시스턴트입니다. 사용자의 말에 공감하며 자연스럽게 대화하세요.
과학적 설명이나 문서 검색 없이 일상적인 대화로 응답하세요.

사용자: {question}
어시스턴트:"""

def generate_chitchat(question: str, llm) -> str:
    prompt = CHITCHAT_PROMPT.format(question=question)
    return llm.complete(prompt).text.strip()
```

분류는 `query_rewrite.is_science_question(msg, llm)`으로 수행합니다.  
Qwen3.5-4B가 `<think>...</think>` 블록을 출력하므로 응답에서 think 블록을 제거한 뒤 yes/no를 판별합니다.

```python
import re

def is_science_question(msg: list[dict], llm) -> bool:
    question = msg[-1]["content"]
    prompt = f"""다음 질문이 과학 상식·지식 관련 질문이면 'yes', 일상 대화이면 'no'로만 답하세요.

질문: {question}
답:"""
    raw = llm.complete(prompt).text.strip()
    # <think>...</think> 블록 제거 후 판별
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    answer = clean.splitlines()[-1].strip().lower() if clean else ""
    if "yes" in answer:
        return True
    if "no" in answer:
        return False
    return True  # 판별 불가 시 과학으로 처리 (안전 방향)
```

**Self-check — Faithfulness 기반 재생성**

> RAGAS Faithfulness 평가자 LLM은 `.env` 키 설정에 따라 자동 선택됩니다.
>
> | 조건 | 사용 LLM |
> |------|----------|
> | `GOOGLE_API_KEY` 설정 | **Gemini 2.0 Flash** (Google AI Studio, 우선) |
> | `OPENAI_API_KEY` 설정 | OpenAI 기본값 |
> | 둘 다 미설정 또는 `DISABLE_SELFCHECK=1` | Self-check 스킵, 첫 번째 생성 결과 반환 |
>
> Google AI Studio는 무료 티어(RPM 15, TPM 1,000,000)를 제공합니다.

```python
import os
from ragas.metrics import faithfulness
from ragas import evaluate
from datasets import Dataset

def _build_ragas_llm():
    """GOOGLE_API_KEY 우선, 없으면 None(ragas OpenAI 기본값)."""
    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper
        return LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.0-flash"))
    return None

def generate_with_selfcheck(question: str, context: str, llm,
                             threshold: float = 0.7,
                             max_retries: int = 2) -> str:
    # API 키 없거나 DISABLE_SELFCHECK=1 이면 단순 생성
    has_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not has_api_key or os.environ.get("DISABLE_SELFCHECK"):
        return generate_answer(question, context, llm)

    answer = generate_answer(question, context, llm)
    retries_left = max_retries
    eval_llm = _build_ragas_llm()
    eval_kwargs = {"llm": eval_llm} if eval_llm else {}

    while True:
        data = Dataset.from_dict({
            "question":  [question],
            "answer":    [answer],
            "contexts":  [[context]],
        })
        # RAGAS 0.4.x: 반환 타입이 버전별로 Dataset / Result 혼재 → 스칼라만 안전 추출
        raw = evaluate(dataset=data, metrics=[faithfulness], **eval_kwargs)
        if hasattr(raw, "to_pandas"):
            score = float(raw.to_pandas()["faithfulness"].iloc[0])
        elif isinstance(raw, dict):
            v = raw["faithfulness"]
            score = float(v[0] if isinstance(v, (list, tuple)) else v)
        else:
            score = float(getattr(raw, "faithfulness", [0.0])[0])

        if score >= threshold:
            return answer

        if retries_left == 0:
            break

        answer = llm.complete(
            "이전 답변이 문서 근거를 충분히 활용하지 못했습니다. "
            "반드시 아래 문서 내용만을 근거로 답하세요.\n\n"
            + SCIENCE_QA_PROMPT.format(context=context, question=question)
        ).text.strip()
        retries_left -= 1

    return answer
```

---

### ⑥ 평가 (Evaluation)

| 항목 | 선택 기술 | 측정 대상 |
|---|---|---|
| 통합 평가 | **RAGAS** | Faithfulness, Answer Relevancy, Context Recall |
| RAGAS 평가 LLM | **Gemini 2.0 Flash** (Google AI Studio, 우선) / OpenAI (대안) | Faithfulness 판정 |
| RAGAS 임베딩 | **text-embedding-004** (Google) / OpenAI (대안) | Answer Relevancy |
| 검색 벤치마크 | BEIR | MAP, MRR, NDCG |
| 실험 추적 | LangSmith | 단계별 레이턴시, 토큰 비용 |

**MAP 평가 — relevance 사전 구축 필수**

eval.jsonl에는 정답 docid가 없으므로 아래 세 방법 중 하나로 relevance를 먼저 구축해야 합니다.

> **로컬 MAP의 한계**: BM25 휴리스틱·Pseudo 라벨로 만든 relevance는 **대회 주최 측 비공개 정답·채점 파이프라인과 수치가 다를 수 있습니다.** 개발 중 상대 비교용으로만 쓰고, 최종 순위는 제출 후 공식 결과를 기준으로 합니다.

| 방법 | 정확도 | 비용 | 권장 단계 |
|---|---|---|---|
| **① BM25 Heuristic** | 낮음 | 무료 | 빠른 기준선 |
| **② Pseudo Labeling** | 중간 | LLM 호출 | **실전 권장** |
| ③ 수동 레이블링 | 높음 | 시간 소요 | 여유 있을 때 |

```python
import json

# ── 방법 ①: BM25 Heuristic ──────────────────────────────────────────────────
def build_relevance_bm25(eval_path: str, es_client, llm=None,
                          top_k: int = 3) -> dict[int, set[str]]:
    relevance = {}
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            sample  = json.loads(line.strip())
            eval_id = sample["eval_id"]
            query   = build_search_query(sample["msg"], llm=llm)

            resp = es_client.search(
                index="science_docs",
                query={"match": {"content": query}},
                size=top_k,
            )
            relevance[eval_id] = {h["_id"] for h in resp["hits"]["hits"]}
    return relevance


# ── 방법 ②: Pseudo Labeling (권장) ──────────────────────────────────────────
# 임베딩은 ③절 HuggingFaceEmbedding 과 동일 경로(레포: ir_rag.embeddings) 권장
def build_relevance_pseudo(eval_path: str, doc_path: str,
                            llm, embed_model, top_k: int = 3) -> dict[int, set[str]]:
    import numpy as np
    from numpy.linalg import norm

    docs = {}
    with open(doc_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            docs[d["docid"]] = d["content"]

    doc_ids = list(docs.keys())
    # 의사 코드: 실제로는 배치 encode 로 VRAM/시간 최적화
    doc_mat = np.stack([embed_model.get_text_embedding(docs[i]) for i in doc_ids])
    doc_mat /= norm(doc_mat, axis=1, keepdims=True) + 1e-12

    relevance = {}
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            sample  = json.loads(line.strip())
            eval_id = sample["eval_id"]
            query   = build_search_query(sample["msg"], llm=llm)

            ideal_answer = llm.complete(
                f"다음 과학 질문에 대한 정확한 답변을 작성하세요.\n질문: {query}\n답변:"
            ).text.strip()

            qv = np.array(embed_model.get_query_embedding(query))
            av = np.array(embed_model.get_query_embedding(ideal_answer))
            blend = qv + av
            blend /= norm(blend) + 1e-12
            sims = doc_mat @ blend
            top_idxs = sims.argsort()[-top_k:][::-1]
            relevance[eval_id] = {doc_ids[i] for i in top_idxs}

    return relevance


# ── evaluate_map ─────────────────────────────────────────────────────────────
def evaluate_map(eval_path: str, retriever,
                 relevance_map: dict[int, set[str]],
                 top_k: int = 20, llm=None) -> float:
    ap_scores = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            sample   = json.loads(line.strip())
            eval_id  = sample["eval_id"]
            relevant = relevance_map.get(eval_id, set())

            if not relevant:
                continue

            query       = build_search_query(sample["msg"], llm=llm)
            ranked_docs = retriever(query, top_k=top_k)

            hits, precision_sum = 0, 0.0
            for rank, doc_id in enumerate(ranked_docs, start=1):
                if doc_id in relevant:
                    hits += 1
                    precision_sum += hits / rank

            ap_scores.append(precision_sum / len(relevant))

    map_score = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    print(f"MAP@{top_k}: {map_score:.4f}  (평가 쿼리: {len(ap_scores)}건)")
    return map_score


# 사용 예시
# relevance = build_relevance_bm25("eval.jsonl", es)           # 기준선
# relevance = build_relevance_pseudo("eval.jsonl", "documents.jsonl", llm)  # 실전
# map_score = evaluate_map("eval.jsonl", retriever, relevance, llm=my_llm)
```

---

## 제출 파이프라인 및 파일 포맷

공식 베이스라인은 [baseline_code/rag_with_elasticsearch.py](../baseline_code/rag_with_elasticsearch.py) 및 [data/sample_submission.csv](../data/sample_submission.csv) 를 따릅니다.

**파일 형식**

- **파일명 관례**: `sample_submission.csv` — 확장자는 `.csv` 이지만 **내용은 UTF-8 JSON Lines** (한 줄에 JSON 객체 하나).
- **행 수**: `eval.jsonl` 과 동일하게 **220줄** (빈 줄 없음).

**한 줄(JSON) 필드 (필수)**

| 필드 | 설명 |
|------|------|
| `eval_id` | 정수 |
| `standalone_query` | 검색에 쓰는 단일 쿼리 문자열 (멀티 턴이면 독립 검색 쿼리; 베이스라인은 LLM tool 인자) |
| `topk` | 관련 문서 `docid` 문자열 배열 (리더보드 채점은 **상위 3개**만 사용) |
| `answer` | 최종 생성 답변 (리더보드 MAP에는 직접 쓰이지 않을 수 있음) |
| `references` | `[{"score": number, "content": string}, ...]` 검색 히트 메타 |

**채점(리더보드) 요지** — [baseline_code/README.md](../baseline_code/README.md) 와 동일:

- End-to-end 답변 자동평가 대신 **적절한 레퍼런스(docid) 추출**에 대해 **변형 MAP** 적용.
- 일반 케이스: 각 질의에 대해 `topk[:3]` 와 정답 doc 집합으로 Average Precision 계산 후 평균.
- **검색이 필요 없는 GT**(과학 상식 질문이 아닌 등): 정답 쪽이 「추출 없음」이면 **빈 `topk` → 만점(1), 문서를 건너 냈으면 0** 처리.

공개 데이터에는 **정답 docid 목록(GT)이 없음** → 로컬에서 동일 점수를 재현하려면 주최 측 GT 또는 자체 `eval_gt.jsonl`(형식: `eval_id`, `relevant_docids`)이 필요합니다.

**스크립트**

- [scripts/export_submission.py](../scripts/export_submission.py): `--placeholder` 로 형식만 채움; 실제 파이프라인은 베이스라인 `answer_question` 과 같은 5필드 dict 로 직렬화.
- [scripts/run_competition_map.py](../scripts/run_competition_map.py): `--submission` + `--gt` 로 `calc_map` 계산 ([src/ir_rag/competition_metrics.py](../src/ir_rag/competition_metrics.py)).
- [scripts/validate_submission.py](../scripts/validate_submission.py): 필수 키 검사.

---

## 파이프라인 오케스트레이션 & 인프라

| 역할 | 선택 | 이유 |
|---|---|---|
| RAG 프레임워크 | **LlamaIndex** | HyDE·RAG-Fusion 내장, QueryPipeline A/B 테스트 |
| 실험 추적 | LangSmith | LangChain 없이 단독 사용 가능 |
| SFT 프레임워크 | **Unsloth** | 1.5× 속도, 50% VRAM 절약 |
| 벡터 DB | **Qdrant** | 하이브리드 검색 내장, LlamaIndex 네이티브 |
| 한국어 형태소 분석 | **MeCab** + Nori | C++ 기반, JVM 불필요 |
| 모델 서빙 | vLLM | SFT 모델 OpenAI 호환 API 서빙 |

> LangChain 미사용 이유: 단순 선형 파이프라인(검색 → 리랭킹 → 생성)에서는 LlamaIndex 단독으로 충분합니다. LangSmith는 `pip install langsmith`로 독립 사용 가능합니다.

**레포 실행 골격 (최소 구현)**

| 스크립트 | 설명 |
|----------|------|
| [docs/OPERATION.md](OPERATION.md) | Elasticsearch + Qdrant 직접 설치 및 기동 가이드 |
| [scripts/index_es.py](../scripts/index_es.py) | `documents.jsonl` → ES Nori 색인 |
| [scripts/index_qdrant.py](../scripts/index_qdrant.py) | Qwen 임베딩 + Qdrant 색인 (GPU·대용량) |
| [scripts/build_user_dict.py](../scripts/build_user_dict.py) | MeCab 기반 사용자 사전 |
| [scripts/run_retrieval_eval.py](../scripts/run_retrieval_eval.py) | BM25 pseudo relevance MAP (로컬 상대비교) |
| [scripts/run_competition_map.py](../scripts/run_competition_map.py) | 대회 변형 MAP (`calc_map` + GT) |
| [scripts/export_submission.py](../scripts/export_submission.py) | `sample_submission.csv` 형식(5필드 JSONL) |
| [scripts/validate_submission.py](../scripts/validate_submission.py) | 제출 행 필수 키 검사 |
| [scripts/smoke_e2e.py](../scripts/smoke_e2e.py) | ES 소규모 서브셋 스모크 |
| [baseline_code/](../baseline_code/) | 공식 ES+임베딩+OpenAI RAG 베이스라인 |
| [src/ir_rag/](../src/ir_rag/) | 전처리·쿼리 재작성·임베딩 팩토리·MAP 유틸 |

---

## 24GB VRAM 관리 — 순차 언로드 전략

```python
import torch, gc

def unload_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"VRAM 해제 완료: {torch.cuda.memory_allocated()/1e9:.1f}GB 사용 중")


# 단계 1: 인덱싱
embed_model = HuggingFaceEmbedding(model_name="Qwen/Qwen3-Embedding-8B", ...)
index_all_documents(embed_model)
unload_model(embed_model)

# 단계 2: 검색
embed_model = HuggingFaceEmbedding(model_name="Qwen/Qwen3-Embedding-8B", ...)
top100_candidates = hybrid_search(query, embed_model)   # ~18GB
unload_model(embed_model)

# 단계 3: 리랭킹
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen3-Reranker-8B", torch_dtype=torch.float16
).cuda()
top20 = rerank(top100_candidates, reranker_model)       # ~18GB
unload_model(reranker_model)

# 단계 4: 생성
llm = LLM(model="./qwen35-4b-science-rag", dtype="float16")   # GGUF ~6GB
answer = llm.generate(build_prompt(top20, query))
```

**VRAM 사용 요약**

| 컴포넌트 | 방식 | VRAM |
|---|---|---|
| Qwen3-Embedding-8B | FP16 | ~18GB |
| Qwen3-Reranker-8B | FP16 | ~18GB |
| Qwen3.5-4B | QLoRA SFT | ~12GB |
| Qwen3.5-4B | bf16 LoRA SFT | ~22GB |
| Qwen3.5-4B | GGUF Q4_K_M 서빙 | ~6GB |

---

## 경진대회 단계별 운영 전략

### 1단계 — 초기 실험

- Ubuntu 초기 설정: MeCab, Elasticsearch Nori, Qdrant, CUDA 확인
- documents.jsonl → Elasticsearch(Nori) + Qdrant 동시 색인
- `build_user_dict_from_corpus()` 실행 → 전문 용어 사전 생성
- `build_relevance_bm25()`로 기준선 수립
- 멀티 턴: 단순 concat vs Few-shot LLM 재작성 MAP 비교
- HyDE 조건부 실행 (hyde_threshold=5.0) MAP 비교
- RAGAS + LangSmith로 병목 파악

### 2단계 — 중반 최적화

- `build_relevance_pseudo()`로 Pseudo Labeling 전환
- Reranker 4B → MAP 검증 후 8B 전환 결정
- Unsloth QLoRA로 Qwen3.5-4B 과학 QA SFT
- Soft Voting 효과 검증 후 도입 결정
- Self-check faithfulness threshold 실험 (0.5 / 0.7 / 0.9)
- diskcache 적용으로 반복 실험 속도 향상

### 3단계 — 최종 제출

- QLoRA → bf16 LoRA 전환 후 최종 SFT 재실행
- SFT 모델 + API 앙상블 (다수결 또는 LLM-as-judge)
- 캐시 초기화 후 최종 MAP 재계산 (`cache.clear()`)
- GGUF Q4_K_M 변환 후 vLLM 서빙
- 최종 MAP 측정 및 제출

---

## 전체 스택 한눈에 보기

```
eval.jsonl (220건: 단일 200 / 멀티턴 20)
    │
    ▼
[쿼리 전처리]
  단일 턴: 마지막 user 발화 그대로
  멀티 턴: Few-shot LLM 재작성 → 품질 검증 → 실패 시 원본 fallback
    │
    ├──────────────────────────────────┐
    ▼ (원본 쿼리, 항상 실행)           ▼ (BM25 score < threshold 일 때만)
[Sparse 검색]                     [Dense 검색 + 조건부 HyDE]
Elasticsearch BM25                 Qwen3-Embedding-8B
+ Nori(mixed) + MeCab 전문 용어 사전 + Qdrant (HNSW) + diskcache
    │                                  │
    └──────────────┬───────────────────┘
                   ▼
           [RRF 병합 k=60]
                   ▼
           [Reranking]
       Qwen3-Reranker-4B (실험) / 8B (최종)
        Top-100 → Top-20
       + Soft Voting (실험 검증 후)
                   ▼
           [LLM 생성]
       Qwen3.5-4B (SFT) / CRAG·CoT 프롬프트
       + Self-check (Faithfulness < threshold → 재생성)
                   ▼
           [MAP 평가]
       relevance_map 사전 구축 (BM25 Heuristic → Pseudo Labeling)
       evaluate_map() + RAGAS + LangSmith

참조: documents.jsonl (4,272건 / docid 단위 / 평균 315자)
```

---

## 추가 고려사항

### 1. 청킹 전략

| 청킹 방식 | 적용 여부 | 이유 |
|---|---|---|
| 고정 길이 / Semantic | 불필요 | 이미 청크 크기(평균 315자) |
| **docid 단위 직접 인덱싱** | **권장** | 문서 1개 = 인덱스 단위 1개 |

> 800자 초과 문서는 18건(0.4%)뿐이므로 해당 문서만 선택적으로 분할해도 충분합니다.

### 2. 멀티 턴 쿼리 처리 전략 비교

| 방법 | 복잡도 | MAP 기대 효과 | 권장 단계 |
|---|---|---|---|
| 마지막 user 발화만 | 낮음 | 기준선 | 실험 시작 |
| 전체 대화 단순 concat | 낮음 | +소폭 | 빠른 개선 |
| **Few-shot LLM 재작성** | **중간** | **+중간** | **1단계부터** |
| HyDE 적용 후 재작성 | 높음 | +높음 | 2단계 이후 |

### 3. 한국어 과학 문서 전처리

```python
import re

def preprocess_science_doc(text: str) -> str:
    text = re.sub(r'\$\$(.+?)\$\$', r'[수식: \1]', text, flags=re.DOTALL)
    text = re.sub(r'\$(.+?)\$', r'[수식: \1]', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'(참고문헌|References|Bibliography).*$', '', text, flags=re.DOTALL)
    return text.strip()
```

### 4. 인덱싱 일관성

Elasticsearch와 Qdrant 두 인덱스에 **동일한 docid**로 색인해야 RRF 병합이 정확히 동작합니다.

```
documents.jsonl
    → preprocess_science_doc()
    → Elasticsearch 색인 (id = docid, Nori 자동 적용)
    → Qdrant 색인   (doc_id = docid, Qwen3-Embedding)
```

### 5. 쿼리 캐싱

```python
import diskcache, hashlib

cache = diskcache.Cache("./rag_cache")

def cache_key(prefix: str, text: str) -> str:
    return f"{prefix}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"

def cached_embed(text: str, embed_fn) -> list[float]:
    key = cache_key("embed", text)
    if key in cache:
        return cache[key]
    result = embed_fn(text)
    cache.set(key, result, expire=86400 * 7)
    return result

def cached_hyde(query: str, llm) -> str:
    key = cache_key("hyde", query)
    if key in cache:
        return cache[key]
    result = llm.complete(
        f"다음 과학 질문에 대한 정확한 설명 문단을 100자 내외로 작성하세요.\n질문: {query}\n설명:"
    ).text.strip()
    cache.set(key, result, expire=86400 * 7)
    return result

def cached_rerank(query: str, candidate_ids: list[str], rerank_fn) -> list[str]:
    key = cache_key("rerank", query + "|".join(sorted(candidate_ids)))
    if key in cache:
        return cache[key]
    result = rerank_fn(query, candidate_ids)
    cache.set(key, result, expire=86400)
    return result
```

> 캐시 초기화: `cache.clear()` 또는 `rm -rf ./rag_cache`
> 최종 제출 전 캐시를 비우고 결과를 재계산하세요.

### 6. 검색 성능 체크리스트

**환경 설정**
- [ ] MeCab + mecab-ko-dic 설치 확인 (`echo "테스트" | mecab`)
- [ ] Elasticsearch Nori 플러그인 설치 확인
- [ ] CUDA + PyTorch CUDA 정합 확인 (`nvidia-smi`)

**데이터 처리**
- [ ] documents.jsonl 4,272건 ES + Qdrant 정상 색인 확인
- [ ] eval.jsonl 멀티 턴 20건 Few-shot 재작성 결과 육안 검토
- [ ] Nori `decompound_mode: mixed` + 전문 용어 사전 등록 확인

**검색**
- [ ] BM25 단독 / Dense 단독 / 하이브리드 / 3축 RRF MAP 비교
- [ ] HyDE 조건부 실행 hyde_threshold 튜닝 (2 / 5 / 10)
- [ ] RRF k값 튜닝 (기본 60, 실험적으로 20~120)

**리랭킹**
- [ ] Reranker 4B vs 8B MAP 비교
- [ ] Top-K 후보 수 튜닝 (50 / 100 / 200)
- [ ] Soft Voting w_reranker 튜닝 (0.6 / 0.7 / 0.8)

**평가**
- [ ] relevance_map 구축 방법 결정 (BM25 기준선 → Pseudo Labeling)
- [ ] Pseudo Labeling 결과 샘플 육안 검토
- [ ] Self-check faithfulness threshold 튜닝 (0.5 / 0.7 / 0.9)
- [ ] 캐시 히트율 확인 (`len(cache)`)

**생성**
- [ ] QLoRA vs bf16 LoRA SFT 후 MAP 비교
- [ ] CRAG vs CoT 프롬프트 MAP 비교
- [ ] Thinking 모드 On/Off 성능 차이 확인
- [ ] 멀티 턴 쿼리 단독 MAP vs 전체 MAP 비교

---

## Qwen 패밀리 통일의 이점

임베딩(`Qwen3-Embedding-8B`), 리랭커(`Qwen3-Reranker-8B`), 생성(`Qwen3.5-4B`)을 Qwen으로 통일 시:

1. **표현 일관성** — 동일 사전학습 기반 공유, 검색-생성 의미 격차 최소화
2. **라이선스 통일** — 전체 스택 Apache 2.0
3. **다국어 통일** — 100+ 언어 일관 지원
4. **운영 단순화** — 단일 벤더로 호환성 관리
