# RAG 파이프라인 개선 계획

> **현재 목표**: MAP 0.85+ 달성 (베이스라인 0.6682 → 참조 팀 0.9288)
> 검색 품질 개선 → LLM 품질 개선 순으로 진행한다.
>
> **갱신 (2026-04-14)**: G-1(`--llm-select`) 리더보드 제출 **MAP=MRR=0.8795** — 목표 0.85+ 달성. B-3b(Reranker QLoRA) 단독 제출 **MAP=MRR=0.2439** (기대 대비 저조; 추론·데이터 점검 대상).

---

## 외부 레퍼런스 분석 (2026-04-13)

동일 대회 1위 팀 솔루션 분석 결과 아래 격차가 확인됨:

| 구성 요소 | 참조 팀 (MAP 0.9288) | 우리 현황 |
|---|---|---|
| **문서 메타 정보** | LLM으로 제목·키워드·요약·카테고리 생성 후 색인 | 원본 content 필드만 색인 |
| **ES 유사도 알고리즘** | LMJelinekMercer | 기본 BM25 (TF-IDF 계열) |
| **ES 동의어 사전** | 사용자 정의 동의어 적용 | 없음 |
| **Reranker Fine-tuning** | 도메인 파인튜닝 완료 | QLoRA 시도(B-3b) — 리더보드 단독 **0.2439** (저조) |
| **LLM 최종 선별** | Reranker 이후 LLM 프롬프트로 최적 문서 재선택 | G-1(`--llm-select`) 적용 — 리더보드 **MAP=MRR=0.8795** |

> **청킹 미적용 결정**: 우리 문서의 content 평균 315자 / 최대 1,230자로 이미 패시지 크기.
> 청킹 시 문맥 절단으로 Reranker 정확도 저하 위험이 높아 제외.

---

## 진행 현황 (2026-04-14 갱신)

| Phase  | 항목                                           | 상태           | 결과                                                               |
| ------ | ---------------------------------------------- | -------------- | ------------------------------------------------------------------ |
| 인프라 | Qdrant 인덱싱                                  | ✅ 완료        | 4,272건 인덱싱 (UUID5 직접 upsert 방식으로 LlamaIndex 버그 우회)   |
| B-1    | Hybrid+Reranker+Faithfulness 데이터 재구축     | ✅ 완료        | 포함 158건 / 제외 62건 (통과율 72%) — **오염 확인, 미사용 결정**  |
| A-1    | documents.jsonl 기반 QA 생성 (2000건)          | ✅ 완료        | 포함 1,747건 / 통과율 87.4% → `artifacts/sft_doc_qa.jsonl` (8.9MB) |
| B-2    | A-1 + B-1 데이터 혼합                          | ✅ 완료        | 1,905건 → `artifacts/sft_data_final.jsonl` (9.2MB)                 |
| C-1    | SFT 학습 (Qwen3.5-4B QLoRA 4-bit)              | ✅ 완료        | `artifacts/qwen35-4b-science-rag` — **오염 데이터 영향으로 폐기**  |
| C-2    | 리더보드 제출 #2 (4B SFT)                      | ✅ 완료        | **MAP=0.3364** (Dense 노이즈 + `<think>` 오염)                     |
| D-1~3  | MAP 하락 원인 분석 + BM25 단독 실험            | ✅ 완료        | 제출 #3 MAP=0.4364 — HyDE 미사용이 핵심 원인                       |
| E-1    | Weighted RRF + HyDE 복원 + 9B LLM 전환        | ✅ 완료        | **MAP=0.3864** — Dense 7:3에서도 오염 지속, 베이스라인 미달       |
| E-5    | BM25-only + HyDE 3축 + 9B LLM                 | ⬜ 미시작      | Dense 완전 배제, 베이스라인 회복 목표                              |
| F-1    | 문서 메타 정보 생성                            | ✅ 완료        | 4,272건 생성 → `artifacts/doc_metadata.jsonl` (2.1MB)              |
| F-2a   | 과학 동의어 사전 생성                          | ✅ 완료        | 233개 규칙 → `artifacts/science_synonyms.txt`                      |
| F-2b   | ES 재인덱싱 (LMJelinekMercer + 동의어 + 메타) | ✅ 완료        | 사용자사전 750개 + 동의어 233개 + 멀티필드 + LMJelinekMercer 적용  |
| G-1    | LLM 최종 문서 선별 (Phase 2.5)                 | ✅ 완료        | 리더보드 **MAP=MRR=0.8795** (`export_submission.py --llm-select`)  |
| B-3a   | Reranker 트리플렛 생성                         | ✅ 완료        | 1,747건 → `artifacts/reranker_triplets.jsonl`                      |
| B-3b   | Reranker Fine-tuning                           | ✅ 완료        | QLoRA 1968 steps / 3 epoch — 리더보드 **MAP=MRR=0.2439** (기대 대비 저조; 점검 대상) |

### 주요 이슈 해결 기록

- **Qdrant 0건 버그**: `index_qdrant.py`를 `uuid.uuid5` 직접 upsert 방식으로 전면 교체
- **B-1 데이터 오염**: Dense 검색 노이즈로 문서-질문 불일치 ~60% → 학습 미사용 결정
- **SFT 4B 모델 폐기**: 오염된 B-1 포함 학습 → Qwen3.5-9B 기본 모델로 전환

---

## 리더보드 실험 기록

| # | 실험 구성 | MAP | MRR | 비고 |
|---|---|---|---|---|
| 1 | 베이스라인 (BM25 + HyDE 3축 + Reranker + 4B LLM) | **0.6682** | 0.6682 | `old/sample_submission_clean_1.csv` |
| 2 | Hybrid(BM25+Dense 균등) + alt_query + Reranker + 4B SFT | **0.3364** | 0.3379 | Dense 노이즈 + `<think>` 오염 |
| 3 | BM25 단독(--skip-dense) + Reranker + 4B SFT | **0.4364** | 0.4379 | HyDE 미사용으로 베이스라인 미달 |
| E-1 | BM25+Dense 7:3 + HyDE 2축 + Reranker + 9B | **0.3864** | 0.3879 | Dense 7:3에서도 오염 지속 |
| B-3b | 파인튜닝 Reranker(QLoRA) 적용 제출 | **0.2439** | 0.2439 | Reranker만 교체; 기대 대비 저조 |
| G-1 | `--llm-select` LLM 최종 문서 선별 | **0.8795** | 0.8795 | 목표 0.85+ 달성 |

### Dense 오염 패턴 분석

| 실험 | Dense | MAP | 비고 |
|---|---|---|---|
| #1 (baseline) | ❌ 없음 | **0.6682** | BM25 3축 |
| #3 | ❌ 없음 | 0.4364 | BM25 1축 (HyDE 미사용) |
| #2 | ✅ 균등 | 0.3364 | Dense 오염 최악 |
| E-1 | ✅ 7:3 | 0.3864 | Dense 비중 줄여도 오염 지속 |

**결론**: Dense 인덱스에 근본적 문제 존재. 가중치 조정으로는 해결 불가.
→ Dense 완전 배제 후 BM25 품질(F-1/F-2) 향상에 집중하는 전략으로 전환.

### 실험 #3 갭 원인 분석 (0.4364 vs 0.6682)

| 항목 | 베이스라인 #1 | 실험 #3 (BM25-only) |
|---|---|---|
| Phase 0 출력 | standalone + hyde_doc + alt_query | standalone만 |
| Phase 1 RRF | 3축 (standalone+HyDE+alt via BM25) | 1축 (standalone만) |
| topk 일치율 (vs #1) | — | 평균 1.13/3 (37%) |
| 완전 불일치 쿼리 | — | 31건 (16.6%) |

---

## Phase E — HyDE 복원 + Weighted RRF

### E-1. Qwen3.5-9B + HyDE 2축 + BM25:Dense 7:3 🔄 진행 중

#### 현재 파이프라인 동작

```
Phase 0: Qwen3.5-9B (no SFT)
  - 치챗/과학 분류, 멀티턴 standalone 재작성
  - 과학 쿼리마다 HyDE 문서 생성 (<think> strip 후 사용)

Phase 1: Weighted RRF (BM25:Dense = 7:3)
  - standalone + HyDE doc → 각각 BM25+Dense 검색
  - 2축 RRF → 후보 top-20

Phase 2: Qwen3-Reranker-8B
  - 후보 top-20 → rerank → top-10

Phase 3: Qwen3.5-9B 답변 생성
```

#### E-1 결과: MAP=0.3864 — Dense 오염 지속 확인

#### E-5. BM25-only + HyDE 3축 + 9B LLM ← 다음 실험

Dense를 완전히 배제하고 HyDE + alt_query 3축 BM25 RRF를 복원한다.
Phase 0 캐시를 재사용하므로 빠르게 실행 가능.

```bash
# phase0 캐시 재사용 (HyDE/alt_query 이미 생성됨)
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --skip-dense \
  --phase0-cache artifacts/phase0_queries.csv \
  --output artifacts/sample_submission_e5_bm25_hyde.csv
```

**예상**: HyDE 3축이 살아있으므로 베이스라인 #1(0.6682) 수준 회복 기대.
9B vs 4B 모델 차이가 있으나 검색 품질이 회복되면 근접할 것으로 판단.

#### 이후 실험 후보 (E-5 결과 확인 후)

| 실험 | 설명 | 조건 |
|---|---|---|
| F-1→F-2 | 메타 색인 + LMJelinekMercer + 동의어 | E-5 MAP ≥ 0.60 이면 착수 |
| G-1 | LLM 최종 문서 선별 (--llm-select) | ✅ 완료 — MAP=MRR=0.8795 (2026-04-14) |
| Dense 재인덱싱 | Qdrant 인덱스 재구축 후 재실험 | 원인 파악 후 별도 검토 |

---

## Phase F — ES·색인 고도화

> E-1 결과 확인 후 F-1 → F-2 순서로 착수.

### F-1. 문서 메타 정보 생성 🔧

**목적**: BM25 색인 품질 향상. 본문에 키워드가 없어도 LLM 생성 요약·키워드로 검색 가능.

**생성 항목**

| 필드 | 형식 | 예시 |
|---|---|---|
| `title` | 30자 이내 | "광합성 명반응의 ATP 합성 과정" |
| `keywords` | 최대 8개 | ["광합성", "ATP", "엽록체", "명반응", "NADPH"] |
| `summary` | 2~3문장 요약 | "이 문서는 ... 를 설명한다." |
| `category` | 분야 | 물리 / 화학 / 생물 / 지구과학 / 기타 |

**스크립트**: `scripts/build_doc_metadata.py`

```bash
# 전체 4,272건 처리 (중단 후 재실행 시 자동 이어받기)
python scripts/build_doc_metadata.py \
  --config config/default.yaml \
  --api solar \
  --output artifacts/doc_metadata.jsonl

# 소규모 테스트 (100건만)
python scripts/build_doc_metadata.py \
  --config config/default.yaml \
  --api solar \
  --max-docs 100 \
  --output artifacts/doc_metadata.jsonl
```

**출력**: `artifacts/doc_metadata.jsonl`
**예상 소요**: Solar API 호출 ~4,272회 (문서당 1회, `--delay 0.1` 기본 적용)

---

### F-2. ES 재인덱싱 (LMJelinekMercer + 동의어 + 멀티필드) 🔧

**목적**: BM25 정밀도·재현율 동시 향상.

#### F-2-a. LMJelinekMercer 유사도

기본 BM25(Okapi BM25)는 짧은 문서에 편향됨. LMJelinekMercer는 문서 길이에 강건.
`--lm-jelinek-mercer` 플래그로 활성화 (λ=0.7 고정).

#### F-2-b. 과학 도메인 동의어 사전

`doc_metadata.jsonl` keywords 빈출 top-300에서 LLM으로 동의어 쌍 추출 → ES synonym filter 적용.

**스크립트**: `scripts/build_synonyms.py`

```bash
python scripts/build_synonyms.py \
  --metadata artifacts/doc_metadata.jsonl \
  --api solar \
  --output artifacts/science_synonyms.txt
```

#### F-2-c. 멀티필드 색인

`content` 단일 필드 → `title^2 / keywords^1.5 / summary / content` 멀티필드 검색.
검색 시 `--multi-field` 플래그로 `multi_match` 쿼리 활성화.

#### F-2 실행 순서

```bash
# 1. F-1 먼저 실행 후

# 2. 동의어 사전 빌드
python scripts/build_synonyms.py \
  --metadata artifacts/doc_metadata.jsonl \
  --api solar \
  --output artifacts/science_synonyms.txt

# 3. ES 인덱스 재생성 (기존 인덱스 삭제 후 새 설정 적용)
python scripts/index_es.py \
  --config config/default.yaml \
  --metadata artifacts/doc_metadata.jsonl \
  --synonyms artifacts/science_synonyms.txt \
  --lm-jelinek-mercer \
  --recreate

# 4. 재인덱싱 후 파이프라인 실행 (멀티필드 검색 활성화)
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --bm25-weight 0.7 --dense-weight 0.3 \
  --multi-field \
  --phase0-cache artifacts/phase0_queries.csv \
  --output artifacts/sample_submission_f2.csv
```

---

## Phase G — LLM 최종 문서 선별 (Phase 2.5)

#### G-1 결과 (2026-04-14)

리더보드 **MAP=MRR=0.8795** — 상단 목표(0.85+) 달성.

**현재**: Reranker top-10 → 답변 생성 (수치 점수만으로 결정)
**개선**: Reranker top-10 → **LLM 선별 top-3** → 답변 생성

**목적**: 수치적 reranker 점수가 놓치는 의미적 관련성을 LLM이 포착.
**VRAM**: Phase 3 LLM과 별개로 로드·언로드하므로 추가 VRAM 불필요.
**폴백**: LLM 파싱 실패 시 Reranker 점수 순 top-3 자동 사용.

#### G-1 파이프라인 동작

```
Phase 2:   Reranker top-10 산출
Phase 2.5: LLM 선별 (--llm-select 활성화 시)
  - top-10 문서를 content[:250] 스니펫과 함께 LLM에 전달
  - LLM이 문서 번호 3개 선택 (예: "1, 3, 7")
  - 파싱 실패 → Reranker 점수 순 top-3 폴백
Phase 3:   선별된 3개 문서로 답변 생성
```

#### G-1 실행 명령

```bash
# Phase 0 캐시 재사용 + LLM 선별 활성화
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --bm25-weight 0.7 --dense-weight 0.3 \
  --phase0-cache artifacts/phase0_queries.csv \
  --llm-select \
  --output artifacts/sample_submission_g1.csv

# F-2 완료 후 멀티필드까지 적용
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --bm25-weight 0.7 --dense-weight 0.3 \
  --multi-field \
  --phase0-cache artifacts/phase0_queries.csv \
  --llm-select \
  --output artifacts/sample_submission_f2_g1.csv
```

---

## Phase B-3 — Reranker Fine-tuning

> G-1 이후 또는 병행 가능. A-1 생성 트리플렛 ~1,747건 분량 이미 확보.

### 학습 데이터 구성

```
sft_doc_qa.jsonl 각 행 →
  user 메시지 파싱 → 질문 + 첫 번째 문서 내용 추출
  첫 번째 문서 content 앞 50자로 documents.jsonl과 매칭 → positive docid
  BM25 검색 → negative docids (positive 제외, 최대 5개)
  → (query, positive, negatives) 트리플렛
```

| 데이터 소스 | positive | negative | 품질 |
|---|---|---|---|
| A-1 source doc | 질문 생성 원본 문서 (구조적 보장) | 동일 쿼리 BM25 검색 결과 나머지 | 양호 |

### 학습 방식

```
모델: Qwen/Qwen3-Reranker-8B
방식: QLoRA 4-bit (bitsandbytes + peft, 24GB VRAM 내 가능)
손실: BCEWithLogitsLoss (positive 쌍=1.0, negative 쌍=0.0)
배치: per_device=2, grad_accum=8 → 유효 배치 16
```

#### B-3b 리더보드 결과 (2026-04-14)

QLoRA 파인튜닝 Reranker(`artifacts/qwen3-reranker-8b-science`) 적용 제출: **MAP=MRR=0.2439** — 기대 대비 매우 낮음. 트리플렛·음성 샘플링·도메인 불일치 및 추론 시 PEFT 어댑터 로드 여부(`load_reranker`) 등 추가 분석 예정.

### 실행 명령

```bash
# 1. 트리플렛 데이터 생성 (ES 연결 필요)
python scripts/build_reranker_triplets.py \
  --config config/default.yaml \
  --sft-data artifacts/sft_doc_qa.jsonl \
  --output artifacts/reranker_triplets.jsonl

# F-2 메타 색인 완료 후 멀티필드 검색으로 재생성 시
python scripts/build_reranker_triplets.py \
  --config config/default.yaml \
  --sft-data artifacts/sft_doc_qa.jsonl \
  --use-multi-field \
  --output artifacts/reranker_triplets.jsonl

# 2. Reranker Fine-tuning (GPU 필요, ~24GB VRAM)
python scripts/train_reranker.py \
  --config config/default.yaml \
  --data artifacts/reranker_triplets.jsonl \
  --model Qwen/Qwen3-Reranker-8B \
  --output-dir artifacts/qwen3-reranker-8b-science \
  --epochs 3

# 3. 파인튜닝된 Reranker 적용 — config/default.yaml 수정 후 파이프라인 재실행
#    reranker.model_name: "artifacts/qwen3-reranker-8b-science"
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --bm25-weight 0.7 --dense-weight 0.3 \
  --phase0-cache artifacts/phase0_queries.csv \
  --output artifacts/sample_submission_ft_reranker.csv

# 4. MAP 비교
python scripts/run_competition_map.py \
  --submission artifacts/sample_submission_ft_reranker.csv
```

### 판단 기준

- MAP 개선 +0.02 이상이면 채택, 미만이면 기본 Qwen3-Reranker-8B 유지

---

## 향후 검토 대상 (현재 적용 보류)

### ColBERT

토큰 수준 세부 매칭으로 정밀도 향상 가능하나 아래 이유로 보류:
- VRAM 추가 필요 (~8GB), 기존 파이프라인과 통합 복잡도 높음
- F·G·B-3 개선 후 MAP이 0.85 미달할 경우 최후 수단으로 재검토

---

## 전체 실행 순서 요약

| 순서 | Phase  | 스크립트                                              | 상태           | 비고                                          |
| ---- | ------ | ----------------------------------------------------- | -------------- | --------------------------------------------- |
| 0    | 인프라 | `index_qdrant.py`                                     | ✅ 완료        | 4,272건 Qdrant 인덱싱                         |
| 1    | A-1    | `build_sft_from_docs.py`                              | ✅ 완료        | 1,747건 / 통과율 87.4%                        |
| 2    | B-1·2  | `build_sft_data.py` → 혼합                            | ✅ 완료        | 1,905건 → `sft_data_final.jsonl`              |
| 3    | C-1    | `train_sft.py`                                        | ✅ 완료        | 4B SFT — 오염으로 폐기, 9B 기본 모델 전환    |
| 4    | E-1    | `export_submission.py`                                | ✅ 완료        | MAP=0.3864 — Dense 오염 지속 확인             |
| 5    | E-5    | `export_submission.py --skip-dense`                   | ⬜ 미시작      | BM25-only + HyDE 3축, 베이스라인 회복 목표    |
| 6    | F-1    | `build_doc_metadata.py`                               | ✅ 완료        | 4,272건 메타 생성 → `doc_metadata.jsonl`      |
| 7    | F-2a   | `build_synonyms.py`                                   | ✅ 완료        | 동의어 233개 규칙 → `science_synonyms.txt`    |
| 8    | F-2b   | `index_es.py --lm-jelinek-mercer --recreate`          | ✅ 완료        | LMJelinekMercer + 동의어 + 멀티필드 적용      |
| 9    | B-3a   | `build_reranker_triplets.py`                          | ✅ 완료        | 1,747건 트리플렛 → `reranker_triplets.jsonl`  |
| 10   | B-3b   | `train_reranker.py`                                   | ✅ 완료        | 리더보드 MAP=MRR=0.2439 — Reranker 단독 교체 제출                |
| 11   | G-1    | `export_submission.py --llm-select`                   | ✅ 완료        | 리더보드 MAP=MRR=0.8795 — 목표 0.85+ 달성                     |
| 12   | D      | `serve_app.py` + `static/index.html`                  | ⬜ 미시작      | 서빙 UI — 전체 검증 완료 후                   |

> **현재 우선순위**: G-1으로 MAP 0.85+ 달성. B-3b 단독(0.2439)은 저조 — Reranker 추론 경로·데이터 점검. E-5(BM25-only + HyDE)는 선택적으로 베이스라인 회복 비교용.

---

## 구현 내역 (2026-04-13)

### 신규 스크립트

| 파일 | 역할 |
|---|---|
| `scripts/build_doc_metadata.py` | F-1: Solar API로 문서별 title·keywords·summary·category 생성, 자동 이어받기 |
| `scripts/build_synonyms.py` | F-2b: 빈출 키워드 → LLM으로 동의어 쌍 추출 → ES synonym 형식 파일 생성 |
| `scripts/build_reranker_triplets.py` | B-3a: sft_doc_qa.jsonl 파싱 → (query, positive, negatives) 트리플렛 생성 |
| `scripts/train_reranker.py` | B-3b: BCEWithLogitsLoss + QLoRA로 Reranker 파인튜닝 |

### 수정된 파일

| 파일 | 변경 내용 |
|---|---|
| `src/ir_rag/es_util.py` | `ES_META_MAPPINGS` 추가, `_build_settings()` LMJelinekMercer·동의어 파라미터, `load_synonyms_file()`, `ensure_index()` 파라미터 확장 |
| `src/ir_rag/retrieval.py` | `es_bm25_doc_ids()` / `es_bm25_top_score()` — `use_multi_field` 파라미터 추가 |
| `scripts/index_es.py` | `--metadata`, `--synonyms`, `--lm-jelinek-mercer` CLI 옵션 추가 |
| `scripts/export_submission.py` | `_llm_select_docs()` + Phase 2.5 삽입, `--llm-select` / `--multi-field` / `--skip-generation` CLI 옵션 추가, `run_pipeline()` 파라미터 확장 |

---

## Phase A — documents.jsonl 기반 Gold SFT 데이터 구축 ✅ 완료

### 완료 결과 (2026-04-12)

| 항목 | 수치 |
|---|---|
| 처리 목표 문서 수 | 2,000건 |
| Faithfulness 통과 | **1,747건** (통과율 87.4%) |
| 출력 파일 | `artifacts/sft_doc_qa.jsonl` (8.9MB) |
| 검색 방식 | BM25+Dense (--skip-reranker), source doc 강제 포함 |

**처리 흐름**

```
documents.jsonl의 문서 d →
  Solar API로 과학 질문 q 생성 (eval.jsonl 스타일 few-shot 8개 참조)
  → q로 Hybrid 검색 → 검색 결과에 d 포함 여부 확인
      포함 O → 검색 결과 문서 세트로 답변 생성
      포함 X → d 강제 포함 (position 0 삽입)하여 답변 생성
  → RAGAS Faithfulness ≥ 0.7 → 포함 / 미달 → 재생성 1회 → 그래도 미달 → 제외
```

---

## Phase D — 서빙 UI ⬜

> 전체 검증 완료 후 진행.

| 구분 | 선택 | 이유 |
|---|---|---|
| Frontend | Vanilla HTML/CSS/JS (단일 파일) | 프레임워크 불필요, 즉시 배포 가능 |
| 실시간 통신 | SSE (Server-Sent Events) | 단방향 스트림에 최적 |
| Backend | FastAPI + `uvicorn` | async 스트리밍 지원 |
| LLM 연결 | vLLM OpenAI 호환 API (`localhost:8000`) | `serve_vllm.py` 재사용 |

**구현 체크리스트**

- [ ] `scripts/serve_app.py` — FastAPI 앱, `/api/chat` SSE 엔드포인트
- [ ] `static/index.html` — 채팅 UI (CSS·JS 내장)
- [ ] Query Rewrite → Hybrid Search → Rerank 파이프라인 연결
- [ ] vLLM 스트리밍 연결 및 SSE 이벤트 전달

---

## 부록 — 자주 쓰는 명령어 모음

```bash
# E-5: BM25-only + HyDE 3축 (베이스라인 회복 확인용)
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --skip-dense \
  --phase0-cache artifacts/phase0_queries.csv \
  --output artifacts/sample_submission_e5_bm25_hyde.csv

# E-5 + F-2 멀티필드 + 생성 생략 (검색 품질만 빠르게 확인)
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --skip-dense --multi-field \
  --phase0-cache artifacts/phase0_queries.csv \
  --skip-generation \
  --output artifacts/sample_submission_f2_retrieval.csv

# Phase 0 캐시 재사용 (빠른 재실험)
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --bm25-weight 0.7 --dense-weight 0.3 \
  --phase0-cache artifacts/phase0_queries.csv \
  --output artifacts/sample_submission_rerun.csv

# G-1 단독 테스트 (F-2 없이도 가능)
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --bm25-weight 0.7 --dense-weight 0.3 \
  --phase0-cache artifacts/phase0_queries.csv \
  --llm-select \
  --output artifacts/sample_submission_g1.csv

# F-2 + G-1 풀 실험
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --bm25-weight 0.7 --dense-weight 0.3 \
  --multi-field --llm-select \
  --phase0-cache artifacts/phase0_queries.csv \
  --output artifacts/sample_submission_f2_g1.csv

# MAP 평가
python scripts/run_competition_map.py \
  --submission artifacts/sample_submission_9b_w73.csv

# 제출 형식 검증
python scripts/validate_submission.py artifacts/sample_submission_9b_w73.csv
```

```bash
# 검색 실험 모드 (LLM 생성 생략 — 빠른 MAP 측정)
# --skip-dense: Dense 오염 확인으로 기본 옵션
# --skip-generation: Phase 3 LLM 로드 생략 → 검색+재순위만 실행
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --skip-dense \
  --phase0-cache artifacts/phase0_queries.csv \
  --skip-generation \
  --output artifacts/sample_submission_retrieval_only.csv

# F-2 완료 후 멀티필드 검색 적용 (생성 생략)
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --skip-dense --multi-field \
  --phase0-cache artifacts/phase0_queries.csv \
  --skip-generation \
  --output artifacts/sample_submission_f2_retrieval.csv
```

```yaml
# config/default.yaml — Reranker 전환 (B-3 파인튜닝 후)
reranker:
  model_name: "artifacts/qwen3-reranker-8b-science"
  trust_remote_code: true
```
