# SFT 데이터 품질 개선 계획

> **목적**: `artifacts/sft_data.jsonl` (200건)의 문서-질문 불일치 문제 해결 및  
> 고품질 SFT 데이터 재구축을 단계별로 수행한다.

---

## 현황 진단

| 항목 | 현재 상태 | 문제 |
|------|----------|------|
| 검색 방식 | `build_sft_data.py` — BM25 단독 | Dense 검색 미사용 |
| 문서 필터 | `rel_score_ratio=0.7`, `min_bm25_score=1.0` | 표층 키워드 매칭으로 노이즈 문서 혼입 |
| 답변 생성 | 외부 API (Solar/Google) | 노이즈 문서 기반 답변이 이미 포함 |
| Reranker 활용 | 파이프라인(export)에서만 사용 | SFT 빌드 시 미사용 |
| 데이터 양 | 200건 | eval.jsonl 기반 상한선 |

**핵심 원인**: `build_sft_data.py`가 BM25만 사용하기 때문에  
"분류"라는 표층 키워드가 겹치는 명왕성 재분류, 생물 재분류 등이 나무 분류 질문에 혼입됨.

---

## 전체 로드맵 (가성비 높은 순)

```
Stage 1  파라미터 재조정 재실행          ← 코드 수정 없음, 즉시 가능
Stage 2  Hybrid 검색 + Reranker 필터   ← build_sft_data.py 확장
Stage 3  RAGAS Faithfulness 게이트     ← ragas_eval.py 활용
Stage 4  documents.jsonl 역방향 생성   ← 신규 스크립트 필요
Stage 5  Positive/Negative 혼합        ← Stage 4 이후 선택 적용
```

---

## Stage 1 — 파라미터 재조정 재실행
> **가성비: ★★★★★** | 코드 수정 없음 | LLM/GPU 불필요

### 목표
현재 `build_sft_data.py`의 파라미터를 강화해 BM25 수준에서 제거 가능한 노이즈를 먼저 걷어낸다.

### 방법

```bash
python scripts/build_sft_data.py \
  --config config/default.yaml \
  --phase0-csv artifacts/phase0_queries.csv \
  --top-k 7 \
  --min-bm25-score 5.0 \      # 기존 1.0 → 5.0 (낮은 관련성 문서 차단)
  --rel-score-ratio 0.5 \     # 기존 0.7 → 0.5 (top-1 대비 50% 미만 제거)
  --min-docs 2 \              # 기존 3 → 2 (필터 후 최소 문서 수 완화)
  --answer-api solar \
  --output artifacts/sft_data_v2.jsonl
```

### 기대 효과
- 명왕성/프랑스 신앙 등 BM25 점수가 낮은 노이즈 문서 상당수 제거
- phase0_csv 활용으로 치챗 질문 자동 제외
- 답변도 정제된 문서 기준으로 재생성

### 한계
- BM25 점수가 높아도 의미적으로 무관한 문서는 잔존 가능
- 데이터 양이 200건에서 줄어들 수 있음

---

## Stage 2 — Hybrid 검색 + Reranker 필터
> **가성비: ★★★★☆** | `build_sft_data.py` 확장 | Qdrant + Reranker 모델 필요 (GPU)

### 목표
BM25 단독에서 **BM25 + Dense RRF → Reranker 점수 필터**로 업그레이드.  
표층 키워드가 아닌 의미 유사도 기반으로 문서를 선별한다.

### 변경 범위
- `scripts/build_sft_data.py`의 `retriever` 함수를  
  `ir_rag/retrieval.py`의 `es_bm25_doc_ids` + `qdrant_dense_doc_ids` + `rrf_score` 조합으로 교체
- `ir_rag/reranker.py`의 `rerank_with_crossencoder`로 각 문서에 관련성 점수 부여
- reranker score 임계값 이하 문서 제거 (권장: 0.3 ~ 0.5)

### 처리 흐름

```
query (standalone, phase0_csv 활용)
  → BM25 top-20 + Dense top-20 → RRF(k=60) → 후보 top-20
  → Reranker 점수 산출
  → score < threshold 문서 제거
  → 남은 문서로 컨텍스트 구성 → API 답변 생성
```

### 신규 CLI 인자 (추가 예정)

```bash
python scripts/build_sft_data.py \
  --hybrid \                       # Hybrid 모드 활성화
  --reranker-threshold 0.3 \       # Reranker score 컷오프
  --top-k-retrieve 20 \
  --top-k-rerank 5 \
  ...
```

### 기대 효과
- 의미적으로 무관한 문서 제거율 대폭 향상
- 나무 분류 질문에 명왕성 문서가 섞이는 문제 해소
- Reranker가 이미 파이프라인에 있으므로 추가 모델 불필요

### 한계
- Reranker 추론 시간 발생 (200건 × 문서수 × 추론)
- Qdrant가 실행 중이어야 함

---

## Stage 3 — RAGAS Faithfulness 게이트
> **가성비: ★★★☆☆** | `ragas_eval.py` 활용 | 외부 LLM API 필요

### 목표
Stage 1~2로 만들어진 SFT 데이터에서  
**답변이 제공된 문서에 근거하지 않는 row를 제거 또는 재생성**한다.

### 방법

```
생성된 sft_data.jsonl 각 row에 대해:
  1. question, answer, contexts 추출
  2. ragas_eval.py (또는 동등 로직)로 Faithfulness 점수 산출
  3. Faithfulness < 0.7 인 row → 재생성 시도 (max 2회)
     재생성 후에도 미달 → 해당 row 제외
```

### 신규 스크립트 (추가 예정)
`scripts/filter_sft_by_faithfulness.py`

```bash
python scripts/filter_sft_by_faithfulness.py \
  --input  artifacts/sft_data_v2.jsonl \
  --output artifacts/sft_data_v3.jsonl \
  --faithfulness-threshold 0.7 \
  --answer-api solar \
  --max-regenerate 2
```

### 기대 효과
- 문서와 무관한 내용을 추측한 답변 제거
- 모델이 "문서 밖 추측 금지" 패턴을 더 명확히 학습

### 한계
- RAGAS가 외부 LLM(Google AI / OpenAI) API 호출 필요
- 200건 기준 API 비용 발생
- Faithfulness 점수 자체도 LLM 판단에 의존하므로 완벽하지 않음

---

## Stage 4 — documents.jsonl 역방향 생성 (Gold Standard)
> **가성비: ★★☆☆☆** | 신규 스크립트 필요 | LLM API 대량 호출

### 목표
`data/documents.jsonl` (4,272건)의 각 문서에서 직접 질문을 생성해  
**문서-질문-답변 삼각 일관성이 설계 단계에서 보장된** 고품질 데이터를 만든다.

### 처리 흐름

```
documents.jsonl의 문서 d 선택
  → LLM으로 d를 근거로 답할 수 있는 질문 q 생성
  → q로 Hybrid 검색 → Reranker top-K
  → 검색 결과에 d가 포함되는지 확인 (관련성 검증)
    포함 O → 해당 문서 세트로 답변 생성
    포함 X → d를 강제 포함하거나 해당 샘플 스킵
  → RAGAS Self-check 통과 시 최종 데이터셋 포함
```

### 신규 스크립트 (추가 예정)
`scripts/build_sft_from_docs.py`

```bash
python scripts/build_sft_from_docs.py \
  --config config/default.yaml \
  --question-api solar \
  --answer-api solar \
  --top-k-retrieve 20 \
  --top-k-rerank 5 \
  --max-docs 500 \             # 전체 4272건 중 샘플링
  --output artifacts/sft_data_gold.jsonl
```

### 기대 효과
- eval.jsonl 200건 한계 탈피, 데이터 양 대폭 확장 가능
- 문서와 질문의 관련성이 구조적으로 보장됨
- Stage 1~3 데이터와 혼합해 학습 데이터 다양성 확보

### 한계
- 문서 1건당 LLM 호출 2회 이상 (질문 생성 + 답변 생성)
- 4,272건 전체 처리 시 API 비용 및 시간 큼 → 샘플링 필요
- 생성된 질문이 실제 eval 질문 스타일과 다를 수 있음

---

## Stage 5 — Positive/Negative 혼합 (Robustness 강화)
> **가성비: ★★☆☆☆** | Stage 4 이후 선택 적용 | 신규 스크립트 필요

### 목표
실제 검색에서 노이즈 문서가 들어와도 **무시하고 관련 문서만 근거로 답하는 능력**을 강화한다.

### 방법

```
Stage 1~4로 만든 clean 데이터 각 row에 대해:
  - Positive 문서: 기존 관련 문서 (유지)
  - Negative 문서: 다른 주제 질문의 top-1 문서 1~2개 (의도적 혼입)
  → 섞인 문서 세트로 답변 재생성
    단, 시스템 프롬프트에 "관련 없는 문서는 무시하라" 지시 추가
```

### 기대 효과
- RAG 파이프라인에서 검색 노이즈 대응 능력 강화
- 모델이 문서 선별 판단력을 학습

### 한계
- 답변 재생성 비용 발생
- 프롬프트 설계가 복잡해짐
- 기본 성능이 충분하면 불필요할 수 있음

---

## 단계별 실행 요약

| 단계 | 스크립트 | 신규 개발 | GPU 필요 | API 비용 | 예상 산출 건수 |
|------|---------|----------|---------|---------|------------|
| Stage 1 | `build_sft_data.py` (파라미터만) | 없음 | 없음 | 소 (답변 재생성) | ~150~180건 |
| Stage 2 | `build_sft_data.py` (확장) | 중 | 필요 (Reranker) | 소 | ~150~180건 |
| Stage 3 | `filter_sft_by_faithfulness.py` | 소 | 없음 | 중 | Stage 2 결과 필터 |
| Stage 4 | `build_sft_from_docs.py` | 대 | 필요 (Reranker) | 대 | 수백~수천 건 |
| Stage 5 | (미정) | 중 | 없음 | 중 | Stage 4 결과 증강 |

> **권장 진행 순서**: Stage 1 → 2 → 3 순으로 적용 후 품질 확인.  
> 데이터 양이 부족하다고 판단되면 Stage 4 진행.  
> Stage 5는 최종 모델 성능이 기대에 미치지 못할 경우 검토.

---

---

# 실시간 서빙 UI 구현 계획

> **목적**: RAG 파이프라인을 웹 UI로 서빙.  
> 사용자가 채팅창에 질문하면 LLM 답변(스트리밍)과 관련 문서 Top-3를 실시간으로 표시한다.

---

## 기술 스택

| 구분 | 선택 | 이유 |
|------|------|------|
| Frontend | Vanilla HTML/CSS/JS (단일 파일) | 프레임워크 불필요, 배포 즉시 가능 |
| 실시간 통신 | SSE (Server-Sent Events) | 단방향 스트림에 최적, WebSocket 대비 구현 단순 |
| Backend | FastAPI + `uvicorn` | 기존 Python 환경 활용, async 스트리밍 지원 |
| LLM 연결 | vLLM OpenAI 호환 API (`localhost:8000`) | `serve_vllm.py` 기존 서버 재사용, `stream=True` 지원 |
| 검색 | 기존 ES + Qdrant + Reranker 모듈 재사용 | 추가 인프라 불필요 |

---

## 디렉터리 구조

```
scripts/
  serve_app.py          # FastAPI 앱 (신규)
static/
  index.html            # 단일 HTML (CSS·JS 내장, 신규)
```

---

## 서비스 플로우

```
[사용자 입력] → POST /api/chat
                    │
                    ├─ 1. Query Rewrite (vLLM)
                    │      standalone query 생성
                    │
                    ├─ 2. Hybrid Search (ES BM25 + Qdrant Dense → RRF)
                    │
                    ├─ 3. Reranker → top-3 doc_id + content[:50]
                    │
                    └─ 4. LLM 스트리밍 (vLLM stream=True)
                               │
                    SSE event: token      → 프론트에서 글자 단위 렌더링
                    SSE event: docs       → 검색 완료 시 top-3 문서 카드 표시
                    SSE event: done       → 완료 신호
                    SSE event: error      → 오류 메시지
```

---

## API 명세

### `POST /api/chat`
**Request**
```json
{
  "query": "사용자 질문",
  "history": [
    {"role": "user",      "content": "이전 질문"},
    {"role": "assistant", "content": "이전 답변"}
  ]
}
```

**Response**: `text/event-stream` (SSE)
```
data: {"type": "token",  "content": "답변 토큰..."}
data: {"type": "docs",   "docs": [{"id": "doc001", "snippet": "문서 내용 50자..."}]}
data: {"type": "done"}
data: {"type": "error",  "message": "오류 내용"}
```

---

## 화면 구성 (index.html)

```
┌─────────────────────────────────────────────────┐
│  ◉  Science RAG                    [새 질문] 버튼 │
├─────────────────────────────────────────────────┤
│                                                  │
│  [사용자] 나무 분류 방법은?                        │
│                                                  │
│  [AI] 나무의 분류는 형태적 특징과 분자 수준         │
│       분석을 통해 이루어집니다...  ▌(스트리밍)      │
│                                                  │
│  ┌──────── 참고 문서 ────────────────────────┐   │
│  │ 📄 doc042  한 학생이 다양한 종류의 나무를…  │   │
│  │ 📄 doc017  생물학에서 일부 생물체의 분류…   │   │
│  │ 📄 doc089  같은 속에 속한 나무들은 종류…   │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
├─────────────────────────────────────────────────┤
│  [질문을 입력하세요...]               [전송 ▶]    │
└─────────────────────────────────────────────────┘
```

- 답변: 토큰 스트리밍으로 타이핑 효과
- 참고 문서: 검색 완료 시점(`docs` 이벤트)에 카드 형태로 표시
- `새 질문` 버튼: 히스토리 초기화 후 초기 화면 복귀
- 멀티턴: `history` 배열로 이전 대화 컨텍스트 유지

---

## 구현 세부사항

### Backend (`scripts/serve_app.py`)

```python
# 핵심 구조 (의사코드)
@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def generate():
        # 1. Query Rewrite
        standalone = rewrite(req.query, req.history)

        # 2. Hybrid Search + Rerank → top-3
        docs = hybrid_search_and_rerank(standalone, top_k=3)
        # docs 이벤트: 검색 결과 즉시 전송
        yield f'data: {{"type":"docs","docs":{docs_to_json(docs)}}}\n\n'

        # 3. LLM 스트리밍 (vLLM OpenAI API)
        context = format_context(docs)
        stream = openai_client.chat.completions.create(
            model="science-rag",
            messages=build_messages(req.history, standalone, context),
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield f'data: {{"type":"token","content":{json.dumps(token)}}}\n\n'

        yield 'data: {"type":"done"}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Frontend 핵심 JS 로직

```javascript
// SSE 수신 처리
const es = new EventSource('/api/chat');  // 실제론 fetch + ReadableStream 사용
source.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (data.type === 'token')  appendToken(data.content);   // 글자 추가
  if (data.type === 'docs')   renderDocCards(data.docs);   // 문서 카드 표시
  if (data.type === 'done')   enableInput();               // 입력 활성화
  if (data.type === 'error')  showError(data.message);
};
```

> SSE는 GET 방식만 지원하므로 실제 구현 시  
> `fetch` + `ReadableStream` 조합으로 POST 스트리밍 처리

---

## 실행 방법 (구현 완료 후)

```bash
# 1. vLLM 서버 기동 (별도 터미널)
python scripts/serve_vllm.py

# 2. UI 서버 기동
python scripts/serve_app.py --config config/default.yaml --port 7860

# 3. 브라우저 접속
http://localhost:7860
```

---

## 구현 체크리스트

- [ ] `scripts/serve_app.py` — FastAPI 앱, `/api/chat` SSE 엔드포인트
- [ ] `static/index.html` — 채팅 UI (CSS·JS 내장)
- [ ] Query Rewrite → Hybrid Search → Rerank 파이프라인 연결
- [ ] vLLM 스트리밍 연결 및 SSE 이벤트 전달
- [ ] `새 질문` 버튼: 히스토리 초기화
- [ ] 문서 카드: doc_id + content 50자 truncation
- [ ] 오류 처리 (vLLM 미기동, ES/Qdrant 미응답 등)
