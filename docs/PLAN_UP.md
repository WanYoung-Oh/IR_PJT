# RAG 파이프라인 개선 계획

> **현재 목표**: MAP 0.85+ 달성 (베이스라인 0.6682 → 참조 팀 0.9288)
> 검색 품질 개선 → LLM 품질 개선 순으로 진행한다.
>
> **갱신 (2026-04-19)**: ES 사용자 사전·동의어 보완(F-2c) + Embedding Instruction 도메인 강화 후 Qdrant 재인덱싱(F-3) + multi-field boost 튜닝(`Title^2, Keywords^1.5, Summary^1.2, Content^3.5`) → 리더보드 **MAP=0.9250 / MRR=0.9273** (실험 **G-4**, 현재 최고 성능). Reranker SFT 재학습(negatives 오탐 226개 제거, B-3c) 완료 — **결과 미확인**.
>
> **갱신 (2026-04-15)**: **Phase 2.5 비활성화**(`--llm-select` 미사용) — 제출·평가는 **Reranker 출력** 기준. 검색 튜닝: `--multi-field`(멀티필드 boost ^2/^1.5/^1.2/^1.2), BM25:Dense **7:3**, 다축 RRF **`--rrf-weights 0.4,0.3,0.3`**, **`--top-k-retrieve 30`**, **`--top-k-rerank 15`** — 리더보드 **MAP=0.925 / MRR=0.9242** (실험 **G-3**).
>
> **갱신 (2026-04-14)**: G-1(`--llm-select`) 리더보드 제출 **MAP=MRR=0.8795** — 목표 0.85+ 달성. B-3b(Reranker QLoRA) 단독 제출 **MAP=MRR=0.2439** (기대 대비 저조; 추론·데이터 점검 대상).
>
> **갱신 (2026-04-14)**: Phase 0 캐시 `artifacts/phase0_queries_llm_select_total_solar.csv` + Phase 1 다축 RRF 가중치 **`--rrf-weights 0.5,0.25,0.25`** (standalone / HyDE / alt) 적용 제출 — 리더보드 **MAP=0.8386 / MRR=0.8424**. (G-1 대비 설정·캐시가 다르면 수치 직접 비교는 참고용.)
>
> **갱신 (2026-04-14)**: 다축 RRF **`--rrf-weights 0.6,0.15,0.25`** (standalone 비중 확대) 제출 — 리더보드 **MAP=MRR=0.5652**. G-2(0.5/0.25/0.25) 대비 하락. 후속으로 standalone 비중을 줄이고 HyDE·alt에 조금 더 두는 조합(예: **0.4, 0.3, 0.3**) 실험이 유효할 수 있음.

---

## 외부 레퍼런스 분석 (2026-04-13)

동일 대회 1위 팀 솔루션 분석 결과 아래 격차가 확인됨:

| 구성 요소                | 참조 팀 (MAP 0.9288)                           | 우리 현황                                              |
| ------------------------ | ---------------------------------------------- | ------------------------------------------------------ |
| **문서 메타 정보**       | LLM으로 제목·키워드·요약·카테고리 생성 후 색인 | 원본 content 필드만 색인                               |
| **ES 유사도 알고리즘**   | LMJelinekMercer                                | 기본 BM25 (TF-IDF 계열)                                |
| **ES 동의어 사전**       | 사용자 정의 동의어 적용                        | 없음                                                   |
| **Reranker Fine-tuning** | 도메인 파인튜닝 완료                           | QLoRA 시도(B-3b) — 리더보드 단독 **0.2439** (저조)     |
| **LLM 최종 선별**        | Reranker 이후 LLM 프롬프트로 최적 문서 재선택  | G-1(`--llm-select`) 적용 — 리더보드 **MAP=MRR=0.8795** |

> **청킹 미적용 결정**: 우리 문서의 content 평균 315자 / 최대 1,230자로 이미 패시지 크기.
> 청킹 시 문맥 절단으로 Reranker 정확도 저하 위험이 높아 제외.

---

## 진행 현황 (2026-04-15 갱신)

| Phase  | 항목                                                                     | 상태    | 결과                                                                                       |
| ------ | ------------------------------------------------------------------------ | ------- | ------------------------------------------------------------------------------------------ |
| 인프라 | Qdrant 인덱싱                                                            | ✅ 완료 | 4,272건 인덱싱 (UUID5 직접 upsert 방식으로 LlamaIndex 버그 우회)                           |
| B-1    | Hybrid+Reranker+Faithfulness 데이터 재구축                               | ✅ 완료 | 포함 158건 / 제외 62건 (통과율 72%) — **오염 확인, 미사용 결정**                           |
| A-1    | documents.jsonl 기반 QA 생성 (2000건)                                    | ✅ 완료 | 포함 1,747건 / 통과율 87.4% → `artifacts/sft_doc_qa.jsonl` (8.9MB)                         |
| B-2    | A-1 + B-1 데이터 혼합                                                    | ✅ 완료 | 1,905건 → `artifacts/sft_data_final.jsonl` (9.2MB)                                         |
| C-1    | SFT 학습 (Qwen3.5-4B QLoRA 4-bit)                                        | ✅ 완료 | `artifacts/qwen35-4b-science-rag` — **오염 데이터 영향으로 폐기**                          |
| C-2    | 리더보드 제출 #2 (4B SFT)                                                | ✅ 완료 | **MAP=0.3364** (Dense 노이즈 + `<think>` 오염)                                             |
| D-1~3  | MAP 하락 원인 분석 + BM25 단독 실험                                      | ✅ 완료 | 제출 #3 MAP=0.4364 — HyDE 미사용이 핵심 원인                                               |
| E-1    | Weighted RRF + HyDE 복원 + 9B LLM 전환                                   | ✅ 완료 | **MAP=0.3864** — Dense 7:3에서도 오염 지속, 베이스라인 미달                                |
| F-1    | 문서 메타 정보 생성                                                      | ✅ 완료 | 4,272건 생성 → `artifacts/doc_metadata.jsonl` (2.1MB)                                      |
| F-2a   | 과학 동의어 사전 생성                                                    | ✅ 완료 | 233개 규칙 → `artifacts/science_synonyms.txt`                                              |
| F-2b   | ES 재인덱싱 (LMJelinekMercer + 동의어 + 메타)                            | ✅ 완료 | 사용자사전 750개 + 동의어 233개 + 멀티필드 + LMJelinekMercer 적용                          |
| G-1    | LLM 최종 문서 선별 (Phase 2.5)                                           | ✅ 완료 | 리더보드 **MAP=MRR=0.8795** (`export_submission.py --llm-select`)                          |
| G-2    | Solar Phase0 캐시 + 다축 RRF 가중 (0.5/0.25/0.25)                        | ✅ 완료 | 리더보드 **MAP=0.8386 / MRR=0.8424** (`--phase0-cache` + `--rrf-weights`)                  |
| G-3    | 검색·리랭크 파라미터 튜닝 (Phase 2.5 끔)                                 | ✅ 완료 | 리더보드 **MAP=0.925 / MRR=0.9242** — Reranker 출력으로 제출·평가 (`--llm-select` 없음)    |
| B-3a   | Reranker 트리플렛 생성                                                   | ✅ 완료 | 1,747건 → `artifacts/reranker_triplets.jsonl`                                              |
| B-3b   | Reranker Fine-tuning                                                     | ✅ 완료 | QLoRA 1968 steps / 3 epoch                                                                 |
| H-1    | Phase 0 쿼리 품질 분석 (inspection.csv 57건)                             | ✅ 완료 | HyDE 문체 불일치·standalone 구어체·alt_query 단순 재표현 확인 → 개선 방향 수립 (부록 참조) |
| F-2c   | ES 재인덱싱: 사용자 사전 + 동의어 보완                                   | ✅ 완료 | 주요 검색 오류 해결 목적으로 사용자 사전·동의어 일부 추가 후 재인덱싱                      |
| F-3    | Embedding Instruction 도메인 강화 + Qdrant 재인덱싱                      | ✅ 완료 | 과학 도메인 특화 프롬프트로 강화 후 4,272건 재인덱싱                                       |
| G-4    | Multi-field boost 튜닝 (Title^2, Keywords^1.5, Summary^1.2, Content^3.5) | ✅ 완료 | 리더보드 **MAP=0.9250 / MRR=0.9273** — 현재 최고 성능                                      |
| B-3c   | Reranker SFT 재학습 (negatives 오탐 226개 제거)                          | ✅ 완료 | `reranker_triplets.json` 오탐 정제 후 재학습 — **결과 미확인**                             |

### 주요 이슈 해결 기록

- **Qdrant 0건 버그**: `index_qdrant.py`를 `uuid.uuid5` 직접 upsert 방식으로 전면 교체
- **B-1 데이터 오염**: Dense 검색 노이즈로 문서-질문 불일치 ~60% → 학습 미사용 결정
- **SFT 4B 모델 폐기**: 오염된 B-1 포함 학습 → Qwen3.5-9B 기본 모델로 전환

---

## 리더보드 실험 기록

| #    | 실험 구성                                                                                                                                       | MAP        | MRR        | 비고                                                                                              |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- | ------------------------------------------------------------------------------------------------- |
| 1    | 베이스라인 (BM25 + HyDE 3축 + Reranker + 4B LLM)                                                                                                | **0.6682** | 0.6682     | `old/sample_submission_clean_1.csv`                                                               |
| 2    | Hybrid(BM25+Dense 균등) + alt_query + Reranker + 4B SFT                                                                                         | **0.3364** | 0.3379     | Dense 노이즈 + `<think>` 오염                                                                     |
| 3    | BM25 단독(--skip-dense) + Reranker + 4B SFT                                                                                                     | **0.4364** | 0.4379     | HyDE 미사용으로 베이스라인 미달                                                                   |
| E-1  | BM25+Dense 7:3 + HyDE 2축 + Reranker + 9B                                                                                                       | **0.3864** | 0.3879     | Dense 7:3에서도 오염 지속                                                                         |
| B-3b | 파인튜닝 Reranker(QLoRA) 적용 제출                                                                                                              | **0.2439** | 0.2439     | Reranker만 교체; 기대 대비 저조                                                                   |
| G-1  | `--llm-select` LLM 최종 문서 선별                                                                                                               | **0.8795** | 0.8795     | 목표 0.85+ 달성                                                                                   |
| G-2  | `phase0_queries_llm_select_total_solar.csv` + `--rrf-weights 0.5,0.25,0.25`                                                                     | **0.8386** | **0.8424** | Solar Phase0 캐시; 다축 RRF에서 standalone 가중                                                   |
| RRF  | `--rrf-weights 0.6,0.15,0.25`                                                                                                                   | **0.5652** | **0.5652** | standalone 0.6·HyDE 0.15·alt 0.25; G-2 대비 하락 → 균형 잡힌 가중(예: 0.4/0.3/0.3) 추가 실험 후보 |
| G-3  | `--multi-field` · BM25:Dense 7:3 · `--rrf-weights 0.4,0.3,0.3` · `--top-k-retrieve 30` · `--top-k-rerank 15` · **Phase 2.5 미사용**             | **0.925**  | **0.9242** | `--llm-select` 없음; **제출·평가 top-k는 Reranker 출력** 기준                                     |
| G-4  | G-3 + ES 사용자사전·동의어 보완(F-2c) + Embedding Instruction 강화·재인덱싱(F-3) + boost 튜닝 `Title^2, Keywords^1.5, Summary^1.2, Content^3.5` | **0.9250** | **0.9273** | **현재 최고 성능**                                                                                |

### Dense 오염 패턴 분석

| 실험          | Dense   | MAP        | 비고                        |
| ------------- | ------- | ---------- | --------------------------- |
| #1 (baseline) | ❌ 없음 | **0.6682** | BM25 3축                    |
| #3            | ❌ 없음 | 0.4364     | BM25 1축 (HyDE 미사용)      |
| #2            | ✅ 균등 | 0.3364     | Dense 오염 최악             |
| E-1           | ✅ 7:3  | 0.3864     | Dense 비중 줄여도 오염 지속 |

**결론**: Dense 인덱스에 근본적 문제 존재. 가중치 조정으로는 해결 불가.
→ Dense 완전 배제 후 BM25 품질(F-1/F-2) 향상에 집중하는 전략으로 전환.

### 실험 #3 갭 원인 분석 (0.4364 vs 0.6682)

| 항목                | 베이스라인 #1                      | 실험 #3 (BM25-only) |
| ------------------- | ---------------------------------- | ------------------- |
| Phase 0 출력        | standalone + hyde_doc + alt_query  | standalone만        |
| Phase 1 RRF         | 3축 (standalone+HyDE+alt via BM25) | 1축 (standalone만)  |
| topk 일치율 (vs #1) | —                                  | 평균 1.13/3 (37%)   |
| 완전 불일치 쿼리    | —                                  | 31건 (16.6%)        |

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

#### 이후 실험 후보

| 실험           | 설명                              | 상태                                  |
| -------------- | --------------------------------- | ------------------------------------- |
| G-1            | LLM 최종 문서 선별 (--llm-select) | ✅ 완료 — MAP=MRR=0.8795 (2026-04-14) |
| Dense 재인덱싱 | Qdrant 인덱스 재구축 후 재실험    | 원인 파악 후 별도 검토                |

---

## Phase F — ES·색인 고도화

> E-1 결과 확인 후 F-1 → F-2 순서로 착수.

### F-1. 문서 메타 정보 생성 🔧

**목적**: BM25 색인 품질 향상. 본문에 키워드가 없어도 LLM 생성 요약·키워드로 검색 가능.

**생성 항목**

| 필드       | 형식         | 예시                                           |
| ---------- | ------------ | ---------------------------------------------- |
| `title`    | 30자 이내    | "광합성 명반응의 ATP 합성 과정"                |
| `keywords` | 최대 8개     | ["광합성", "ATP", "엽록체", "명반응", "NADPH"] |
| `summary`  | 2~3문장 요약 | "이 문서는 ... 를 설명한다."                   |
| `category` | 분야         | 물리 / 화학 / 생물 / 지구과학 / 기타           |

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
  --llm-select --phase3-api solar \
  --output artifacts/sample_submission_g1.csv

# F-2 완료 후 멀티필드까지 적용
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --bm25-weight 0.7 --dense-weight 0.3 \
  --multi-field \
  --phase0-cache artifacts/phase0_queries.csv \
  --llm-select --phase3-api solar \
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

| 데이터 소스    | positive                          | negative                        | 품질 |
| -------------- | --------------------------------- | ------------------------------- | ---- |
| A-1 source doc | 질문 생성 원본 문서 (구조적 보장) | 동일 쿼리 BM25 검색 결과 나머지 | 양호 |

### 학습 방식

```
모델: Qwen/Qwen3-Reranker-8B
방식: QLoRA 4-bit (bitsandbytes + peft, 24GB VRAM 내 가능)
손실: BCEWithLogitsLoss (positive 쌍=1.0, negative 쌍=0.0)
배치: per_device=2, grad_accum=8 → 유효 배치 16
```

#### B-3b 리더보드 결과 (2026-04-14)

QLoRA 파인튜닝 Reranker(`artifacts/qwen3-reranker-8b-science`) 적용 제출

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
  --phase3-api solar \
  --output artifacts/sample_submission_ft_reranker.csv

# 4. MAP 비교
python scripts/run_competition_map.py \
  --submission artifacts/sample_submission_ft_reranker.csv
```

### 판단 기준

- MAP 개선 +0.02 이상이면 채택, 미만이면 기본 Qwen3-Reranker-8B 유지

---

---

## Phase F-2c — ES 재인덱싱: 사용자 사전 + 동의어 보완 ✅ 완료

**목적**: 주요 검색 오류 케이스에서 확인된 미등록 전문어·표기 변이 해결.

- 기존 F-2b(사용자사전 750개 + 동의어 233개) 기반에 추가 보완
- 검색 실패 로그 분석으로 누락된 사용자 사전 항목·동의어 규칙 일부 추가
- `--recreate` 플래그로 ES 인덱스 재생성하여 적용

---

## Phase F-3 — Embedding Instruction 도메인 강화 + Qdrant 재인덱싱 ✅ 완료

**목적**: Qwen3-Embedding-8B의 Instruction 프롬프트를 과학 교과서 도메인에 최적화하여 Dense 검색 품질 향상.

**변경 내용**

- Instruction 프롬프트에 도메인 컨텍스트 명시 (한국어 중고등학교 수준 과학 문서 대상)
- 변경된 Instruction으로 4,272건 전체 재인덱싱

---

## Phase G-4 — Multi-field Boost 튜닝 ✅ 완료

**목적**: F-2c + F-3 재인덱싱 이후 멀티필드 boost 가중치 재조정으로 검색 정밀도 향상.

**적용 boost 값**

| 필드       | 이전 (G-3) | 변경 (G-4)      |
| ---------- | ---------- | --------------- |
| `title`    | ^2         | **^2** (유지)   |
| `keywords` | ^1.5       | **^1.5** (유지) |
| `summary`  | ^1.2       | **^1.2** (유지) |
| `content`  | ^1.2       | **^3.5** (상향) |

**결과**: 리더보드 **MAP=0.9250 / MRR=0.9273** — 현재 최고 성능

---

## Phase B-3c — Reranker SFT 재학습 (negatives 정제) ✅ 완료 / 결과 미확인

**목적**: B-3b(MAP=MRR=0.2439) 저조 원인 중 하나인 negatives 오탐 제거 후 재학습.

**데이터 정제 내용**

- `reranker_triplets.json`의 negatives에서 실제로는 relevant한 doc_id 오탐 **226개 제거**
- 정제 후 트리플렛 품질 향상 기대

**현재 상태**: 재학습 완료, 리더보드 제출 미시행 — 결과 미확인

---

## 향후 검토 대상 (현재 적용 보류)

### ColBERT

토큰 수준 세부 매칭으로 정밀도 향상 가능하나 아래 이유로 보류:

- VRAM 추가 필요 (~8GB), 기존 파이프라인과 통합 복잡도 높음
- F·G·B-3 개선 후 MAP이 0.85 미달할 경우 최후 수단으로 재검토

---

## 전체 실행 순서 요약

| 순서 | Phase  | 스크립트                                                     | 상태      | 비고                                                                           |
| ---- | ------ | ------------------------------------------------------------ | --------- | ------------------------------------------------------------------------------ |
| 0    | 인프라 | `index_qdrant.py`                                            | ✅ 완료   | 4,272건 Qdrant 인덱싱                                                          |
| 1    | A-1    | `build_sft_from_docs.py`                                     | ✅ 완료   | 1,747건 / 통과율 87.4%                                                         |
| 2    | B-1·2  | `build_sft_data.py` → 혼합                                   | ✅ 완료   | 1,905건 → `sft_data_final.jsonl`                                               |
| 3    | C-1    | `train_sft.py`                                               | ✅ 완료   | 4B SFT — 오염으로 폐기, 9B 기본 모델 전환                                      |
| 4    | E-1    | `export_submission.py`                                       | ✅ 완료   | MAP=0.3864 — Dense 오염 지속 확인                                              |
| 6    | F-1    | `build_doc_metadata.py`                                      | ✅ 완료   | 4,272건 메타 생성 → `doc_metadata.jsonl`                                       |
| 7    | F-2a   | `build_synonyms.py`                                          | ✅ 완료   | 동의어 233개 규칙 → `science_synonyms.txt`                                     |
| 8    | F-2b   | `index_es.py --lm-jelinek-mercer --recreate`                 | ✅ 완료   | LMJelinekMercer + 동의어 + 멀티필드 적용                                       |
| 9    | B-3a   | `build_reranker_triplets.py`                                 | ✅ 완료   | 1,747건 트리플렛 → `reranker_triplets.jsonl`                                   |
| 10   | B-3b   | `train_reranker.py`                                          | ✅ 완료   | 리더보드 MAP=MRR=0.2439 — Reranker 단독 교체 제출                              |
| 11   | G-1    | `export_submission.py --llm-select`                          | ✅ 완료   | 리더보드 MAP=MRR=0.8795 — 목표 0.85+ 달성                                      |
| 12   | G-2    | `export_submission.py` `--rrf-weights` + Phase0 Solar 캐시   | ✅ 완료   | 리더보드 MAP=0.8386 / MRR=0.8424 (`phase0_queries_llm_select_total_solar.csv`) |
| 13   | G-3    | `export_submission.py` 검색·리랭크 튜닝, `--llm-select` 없음 | ✅ 완료   | 리더보드 MAP=0.925 / MRR=0.9242 — Phase 2.5 비활성, Reranker 출력 기준         |
| 14   | F-2c   | `index_es.py --recreate` (사용자사전·동의어 보완)            | ✅ 완료   | 주요 검색 오류 해결 목적                                                       |
| 15   | F-3    | `index_qdrant.py` (Embedding Instruction 강화 후 재인덱싱)   | ✅ 완료   | 과학 도메인 특화 프롬프트 적용                                                 |
| 16   | G-4    | `export_submission.py` boost 튜닝 (Content^3.5)              | ✅ 완료   | 리더보드 **MAP=0.9250 / MRR=0.9273** — 현재 최고 성능                          |
| 17   | B-3c   | `train_reranker.py` (negatives 오탐 226개 제거 후)           | ✅ 완료   | 결과 미확인 — 리더보드 제출 필요                                               |
| 18   | D      | `serve_app.py` + `static/index.html`                         | ⬜ 미시작 | 서빙 UI — 전체 검증 완료 후                                                    |

> **현재 우선순위**: **G-4**가 MAP=0.9250 / MRR=0.9273으로 현재 최고 성능. **B-3c**(Reranker 재학습) 리더보드 제출로 효과 확인 필요. G-3(0.925/0.9242) · G-1(`--llm-select`, 0.8795) · B-3b 단독(0.2439 저조) 순.

---

## 구현 내역 (2026-04-13)

### 신규 스크립트

| 파일                                 | 역할                                                                        |
| ------------------------------------ | --------------------------------------------------------------------------- |
| `scripts/build_doc_metadata.py`      | F-1: Solar API로 문서별 title·keywords·summary·category 생성, 자동 이어받기 |
| `scripts/build_synonyms.py`          | F-2b: 빈출 키워드 → LLM으로 동의어 쌍 추출 → ES synonym 형식 파일 생성      |
| `scripts/build_reranker_triplets.py` | B-3a: sft_doc_qa.jsonl 파싱 → (query, positive, negatives) 트리플렛 생성    |
| `scripts/train_reranker.py`          | B-3b: BCEWithLogitsLoss + QLoRA로 Reranker 파인튜닝                         |

### 수정된 파일

| 파일                           | 변경 내용                                                                                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/ir_rag/es_util.py`        | `ES_META_MAPPINGS` 추가, `_build_settings()` LMJelinekMercer·동의어 파라미터, `load_synonyms_file()`, `ensure_index()` 파라미터 확장        |
| `src/ir_rag/retrieval.py`      | `es_bm25_doc_ids()` / `es_bm25_top_score()` — `use_multi_field` 파라미터 추가                                                               |
| `scripts/index_es.py`          | `--metadata`, `--synonyms`, `--lm-jelinek-mercer` CLI 옵션 추가                                                                             |
| `scripts/export_submission.py` | `_llm_select_docs()` + Phase 2.5 삽입, `--llm-select` / `--multi-field` / `--skip-generation` CLI 옵션 추가, `run_pipeline()` 파라미터 확장 |

---

## Phase A — documents.jsonl 기반 Gold SFT 데이터 구축 ✅ 완료

### 완료 결과 (2026-04-12)

| 항목              | 수치                                               |
| ----------------- | -------------------------------------------------- |
| 처리 목표 문서 수 | 2,000건                                            |
| Faithfulness 통과 | **1,747건** (통과율 87.4%)                         |
| 출력 파일         | `artifacts/sft_doc_qa.jsonl` (8.9MB)               |
| 검색 방식         | BM25+Dense (--skip-reranker), source doc 강제 포함 |

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

| 구분        | 선택                                    | 이유                              |
| ----------- | --------------------------------------- | --------------------------------- |
| Frontend    | Vanilla HTML/CSS/JS (단일 파일)         | 프레임워크 불필요, 즉시 배포 가능 |
| 실시간 통신 | SSE (Server-Sent Events)                | 단방향 스트림에 최적              |
| Backend     | FastAPI + `uvicorn`                     | async 스트리밍 지원               |
| LLM 연결    | vLLM OpenAI 호환 API (`localhost:8000`) | `serve_vllm.py` 재사용            |

**구현 체크리스트**

- [ ] `scripts/serve_app.py` — FastAPI 앱, `/api/chat` SSE 엔드포인트
- [ ] `static/index.html` — 채팅 UI (CSS·JS 내장)
- [ ] Query Rewrite → Hybrid Search → Rerank 파이프라인 연결
- [ ] vLLM 스트리밍 연결 및 SSE 이벤트 전달

---

## 부록 — Phase 0 쿼리 품질 분석 및 개선 방향 (2026-04-17)

> **분석 대상**: `artifacts/inspection.csv` — 검색 실패 케이스 57건  
> **데이터 특성**: 문서 코퍼스는 `ko_mmlu__*` / `ko_ai2_arc__*` 기반 4,272건 (중고등학교 수준 과학 평서문)

### 핵심 발견 요약

| 항목                              | 수치          | 해석                                 |
| --------------------------------- | ------------- | ------------------------------------ |
| 실패 케이스 수                    | 57건          | inspection.csv 전체                  |
| top1 reranker score ≥ 0.95        | 44/57 (77%)   | 문서 자체는 찾았지만 topk 랭킹 문제  |
| 대화체 표현 그대로 사용           | 51/57 (89%)   | standalone이 검색 최적화 미반영      |
| HyDE vs 실제 문서 Jaccard overlap | 0.009 ~ 0.042 | 문체·어휘 괴리 매우 큼               |
| 멀티턴 케이스                     | 7/57 (12%)    | 지시어 미치환·assistant 발화 노이즈  |
| 스토리/사례형 실제 문서           | 9/57 (16%)    | HyDE가 개념 설명형으로 생성돼 불일치 |

---

### 문제 1: HyDE 문체·스타일 불일치 (최우선 개선)

**현상**: HyDE가 마크다운(\*_, _, 헤더), 전문 학술 용어, 고유명사를 포함하여 생성되지만  
실제 문서는 `"~합니다/~입니다"` 체의 간결한 과학 교과서 평서문.

**예시**

```
HyDE:    "**촉매변환기(三元觸媒)**: 배기가스 내 CO, HC, NOx를 CO₂, H₂O, N₂로 변환..."
실제문서: "많은 자동차는 자동차 배기가스에서 탄화수소와 산화물을 제거하는 데 도움을 주는
          촉매 변환 장치가 장착되어 있습니다. 이 장치는 스모그의 생산을 감소시킵니다."
```

**개선 Prompt Instruction**:

```
아래 질문에 답하는 가상의 참조 문서 단락을 작성하세요.

[작성 규칙]
1. 문체: 한국어 중고등학교 과학 교과서 스타일 ("~합니다", "~입니다" 체)
2. 형식: 핵심 개념 → 작동 원리/특징 → 결과/응용 순서, 2~3문장 (100~150자)
3. 금지: **, *, #, - 등 마크다운 서식 / 구체적 인명·제품명·연도 / 불확실한 수치
4. 개념 중심 작성 (고유명사 지양)

좋은 예: "촉매 변환 장치는 자동차 배기가스에서 탄화수소와 산화물을 제거하는 장치입니다.
         이 장치는 스모그의 생산을 감소시키는 데 도움을 줍니다."
나쁜 예: "**촉매변환기**: CO, HC, NOx를 무해한 CO₂, H₂O, N₂로 변환시키는 장치..."
```

---

### 문제 2: Standalone Query 구어체 미변환

**현상**: 57건 중 51건(89%)이 "~에 대해 알려줘", "~가 뭘까", "~어때" 같은 대화체 표현 그대로 사용.  
핵심 개념어가 뒤에 배치되거나 검색에 불필요한 표현 포함.

**예시**

```
나쁨: "차량의 매연이 발생하지 않게 만드는 장치가 무엇인가?"
좋음: "자동차 배기가스 저감 장치의 종류와 원리"

나쁨: "연구자가 갖추어야 할 태도와 자세가 뭘까?"
좋음: "과학 연구자의 태도와 자세: 호기심, 비판적 사고, 성실성"
```

**개선 Prompt Instruction**:

```
검색 최적화 독립 질의 생성 규칙:

1. 핵심 과학/개념 용어를 질의 앞부분에 배치
2. 구어체/대화체 완전 제거
   - 제거 대상: ~알려줘, ~뭘까, ~어때, ~어떤가요, 그거, 이것
   - 변환 형태: ~이란?, ~의 특징은?, ~의 원리는?, ~의 방법은?
3. 이 데이터셋 문서는 MMLU/ARC 기반 한국어 과학 교육 자료 — 교과서 수준 명사형 질의가 효과적
```

---

### 문제 3: Alt Query 단순 재표현 → 다관점 3축으로 확장

**현상**: alt_query가 standalone의 표현만 바꾼 재표현 1개에 그쳐 BM25·dense 검색에서 새로운 문서를 유도하지 못함.

**개선 Prompt Instruction**:

```
다음 질문에 대해 서로 의미적으로 다른 3가지 검색 질의를 생성하세요.
각 질의는 서로 다른 문서를 유도해야 합니다.

- alt_query_1 (정의형):    핵심 개념의 정의·특징 중심  예: "촉매 변환 장치란 무엇인가?"
- alt_query_2 (메커니즘형): 원리·과정·이유 중심       예: "배기가스 유해 물질 화학 변환 반응 원리"
- alt_query_3 (관련개념형): 연관 개념·맥락 포함        예: "스모그 감소 자동차 오염 물질 저감"

추가 규칙:
- 동일 단어 반복 지양 / 영어 원어 병기 허용 (예: "촉매변환기 catalytic converter")
- BM25 검색을 위한 키워드 포함 고려
```

---

### 문제 4: 멀티턴 대화 — 지시어 미치환 및 assistant 발화 노이즈

**현상**: 7개 멀티턴 케이스 중 일부에서 대명사("그", "이것")가 명시어로 치환되지 않거나  
assistant 발화("네 맞습니다", "네 말씀하세요")가 standalone에 포함되어 노이즈 유발.

```
나쁨 standalone: "같은 독감에 다시 걸리면 빨리 회복이 되더라구? 네 맞습니다. 그 이유가 뭐야?"
좋음 standalone: "동일한 독감 변종에 재감염 시 빠른 회복이 일어나는 면역학적 이유"
```

**개선 Prompt Instruction**:

```
멀티턴 대화 처리 규칙:
1. 이전 user 발화에서 핵심 명사구(주제) 추출
2. 마지막 user 발화의 지시어·대명사를 추출된 명사구로 치환
3. assistant 발화("네", "맞습니다", "말씀하세요" 등) 완전 제거
4. 결과: 대화 히스토리 없이도 검색 가능한 완전한 독립 질의

예시:
  [user] 새로 만든 항생제가 나왔어.  [assistant] 네.  [user] 그 효과 검증 방법은?
  → "새로운 항생제의 효능을 검증하기 위한 임상시험 절차"
```

---

### 개선 우선순위

| 순위  | 항목                                        | 기대 효과                | 영향 케이스 |
| ----- | ------------------------------------------- | ------------------------ | ----------- |
| **1** | HyDE 문체를 MMLU/ARC 스타일로 변경          | Dense 검색 recall 향상   | 57건 전체   |
| **2** | Alt query를 3가지 다관점으로 확장           | BM25+dense 커버리지 향상 | 57건 전체   |
| **3** | Standalone 구어체 → 명사형 변환             | 검색 정밀도 향상         | 51건        |
| **4** | 멀티턴: 지시어 명시화 + assistant 발화 제거 | 멀티턴 정확도 향상       | 7건         |

> 가장 빠른 개선 경로: **`src/ir_rag/query_rewrite.py`의 HyDE 생성 프롬프트 수정** (문체 1건만 바꿔도 dense 검색 품질 즉시 개선 기대)

---

## 부록 — 자주 쓰는 명령어 모음

```bash
# Phase 0 캐시 재사용 (빠른 재실험)
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --bm25-weight 0.7 --dense-weight 0.3 \
  --phase0-cache artifacts/phase0_queries.csv \
  --phase3-api solar \
  --output artifacts/sample_submission_rerun.csv

# G-1 단독 테스트 (F-2 없이도 가능)
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --bm25-weight 0.7 --dense-weight 0.3 \
  --phase0-cache artifacts/phase0_queries.csv \
  --llm-select --phase3-api solar \
  --output artifacts/sample_submission_g1.csv

# F-2 + G-1 풀 실험
python scripts/export_submission.py \
  --pipeline --config config/default.yaml \
  --bm25-weight 0.7 --dense-weight 0.3 \
  --multi-field --llm-select \
  --phase0-cache artifacts/phase0_queries.csv \
  --phase3-api solar \
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
