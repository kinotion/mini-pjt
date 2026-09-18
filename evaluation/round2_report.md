# 2차 자체 평가 리포트 (Round 2)

- **평가일**: 2026-09-18
- **대상**: 키즈밀플래너 에이전트의 RAG 파이프라인 정량 평가 + 1차 이후 아키텍처 변경 사항
- **모델**: `global.anthropic.claude-sonnet-4-5-20250929-v1:0` (judge LLM), `amazon.titan-embed-text-v2:0` (임베딩)
- **테스트 셋**: `evaluation/run_ragas.py`의 질문 8건 (`data/knowledge/*.md` 8개 문서에 1:1 대응, ground truth 포함)

## 0. 이번 라운드의 범위 — 먼저 정직하게 밝힌다

**이번 라운드는 `test_queries.csv` 20건의 전체 재실행이 아니다.** [round1_report.md](round1_report.md) 6절이 제안한 4가지 다음 과제(HITL 2턴 실측, 자동 채점 스크립트, CSV `expected_tools` 재검토, 실용성 후속 확인) 중 실제로 착수한 것은 없다 — 전부 3절에서 "미착수"로 그대로 이월한다.

대신 이번 라운드에서 실제로 한 일은 **RAG 파이프라인에 RAGAS 정량 지표를 처음으로 붙인 것**과, 그 과정에서 발생한 아키텍처 변경(Observability를 LangSmith에서 로컬 파일 기반으로 교체) 두 가지다. 1차 리포트가 "행동 판정(pass/fail)"을 다뤘다면, 2차는 "RAG 품질의 숫자 지표"를 다룬다 — 같은 시스템의 다른 단면을 본 것이지, 1차의 후속 검증이 아니다.

## 1. RAGAS 정량 평가 결과

### 1.1 방법론

`data/knowledge/*.md` 8개 문서 각각에 대응하는 질문 1개씩(총 8건)을 실제 에이전트(`agent.handle_query`)로 돌려 답변과 근거 문서(`contexts`)를 얻었다. 질문마다 문서 내용을 그대로 요약한 ground truth를 직접 작성해 `reference`로 넣었다(지어낸 사실 없음 — `evaluation/run_ragas.py`의 `CASES` 참고).

judge LLM과 임베딩은 새 외부 서비스를 쓰지 않고 프로젝트가 이미 쓰는 Bedrock 리소스를 그대로 재사용했다: judge는 `agent.llm`(Claude Sonnet), 임베딩은 `amazon.titan-embed-text-v2:0`. `ragas.evaluate()`에 `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper`로 감싸 넘겼다.

### 1.2 결과 — 표본 크기를 그대로 밝힌다

| 지표 | 확보된 표본 | 값 |
|---|---|---|
| `context_recall` | 4 / 8 | 1.0 |
| `context_precision` | 2 / 8 | 1.0 |
| `faithfulness` | 0 / 8 (배치 기준) | 미확보 — 별도 1문항 사전 테스트에서 **0.9048** 실측 |
| `answer_relevancy` | 0 / 8 | 미확보 |

원본은 [ragas_results.json](ragas_results.json)에 질문별로 남아 있다.

**8건 전부를 확보하지 못한 이유**: 이 프로젝트 전체에서 반복적으로 겪은 Bedrock 계정의 "하루 토큰 총량(Too many tokens per day)" 한도에 걸렸다. RAGAS는 지표 하나당 judge LLM을 여러 번(문항당 1~3회) 호출하므로, 4개 지표 × 8문항이 순식간에 수십 건의 추가 호출을 발생시켰다.

**대응했지만 안 통한 것**: 동시 실행 수를 기본값(`max_workers=16`)에서 `max_workers=1`(완전 순차)로 낮추고 재시도 대기시간도 짧게 줄여 재시도했다. 결과는 거의 동일하게 실패했다 — 즉 이 한도는 초당/분당 요청 제한이 아니라 **진짜 일일 총량 캡**이고, 요청을 늦춰 보내는 방식으로는 우회되지 않는다는 걸 확인했다. 이것 자체가 이번 라운드의 실질적인 발견이다.

## 2. 그 밖의 변경 사항 — Observability

RAGAS 작업 도중, 이 프로젝트의 트레이스 방식을 LangSmith(외부 서버)에서 로컬 파일 기반(`src/local_tracer.py`의 `FileTracer`)으로 교체했다. 이유는 외부 서버 의존을 원치 않는다는 요구 때문이며, 교체 후 `graph.invoke`의 `config={"callbacks": [...]}`을 통해 LLM 호출·도구 호출이 전부 `logs/trace.jsonl`에 기록되는 것을 가짜 모델로 실제 검증했다. RAGAS 지표 자체와는 무관하지만, "Observability·Trace" 패턴의 최종 구현체가 1차 리포트 작성 시점과 달라졌다는 점은 기록해 둔다.

## 3. Round 1 후속 과제 진행 현황

| # | round1_report.md의 제안 | 진행 상태 |
|---|---|---|
| 1 | `id=6`을 같은 스레드로 2턴(식단표 생성 → 저장 요청) 구성해 HITL 승인 흐름 실측 | **미착수** — 3라운드로 이월 |
| 2 | `check_allergen`/`search_recipe` 호출 여부 자동 채점 스크립트 작성 | **미착수** — 이번 라운드에서 RAGAS 스크립트(`run_ragas.py`)는 만들었지만, 이건 RAG 지표용이지 도구 호출 자동 채점용이 아니다. 3라운드로 이월 |
| 3 | `id=12`, `id=19`의 `expected_tools` 설정이 실제 의도에 맞는지 CSV 재검토 | **미착수** — 3라운드로 이월 |
| 4 | id=17 수정 이후 "안전하지만 소극적으로 변한" 응답 성향이 실용성을 해치는지 확인 | **부분 진행** — id=17 재검증 시 모델이 구체적 메뉴 제안 없이 되묻는 쪽으로 수렴한 것을 확인했고(round1_report.md 3절), 안전을 위한 의도된 트레이드오프로 채택했다. 다만 이게 다른 케이스에도 일반적으로 나타나는 경향인지는 20건 전체를 다시 돌려봐야 알 수 있다 — 3라운드로 이월 |

**요약**: 1차가 제안한 4개 과제 중 실제로 마무리된 것은 없다. 이번 라운드는 그 대신 RAG 품질이라는 별도 축을 열었을 뿐이다 — 다음 라운드는 이 4개 과제부터 정리하는 게 우선이다.

## 4. 트라이앤에러 회고

RAGAS를 실제로 붙이는 과정에서 예상보다 훨씬 많은 버전 호환성 문제를 만났고, 그걸 그대로 기록한다.

- **`scikit-network` 빌드 실패 → 해결.** `pip install ragas`가 core 의존성인 `scikit-network`를 소스 빌드하려다 "Microsoft Visual C++ 14.0 필요" 에러로 실패했다. `scikit-network`를 먼저 단독 설치(`--only-binary :all:`)하니 사전 빌드된 wheel이 잡혀서 해결됐다 — `pip`의 의존성 해석 순서 문제였던 것으로 보인다.
- **`ragas`가 이미 삭제된 `langchain_community.chat_models.vertexai`를 무조건 import → 우회.** ragas 0.4.3의 `ragas/llms/base.py`가 이 모듈을 top-level에서 import하는데, 우리 `langchain-community==0.4.2`에는 이 경로가 아예 없다(Vertex AI 지원이 다른 패키지로 이관됨). 우리는 Vertex AI를 전혀 안 쓰므로, `sys.modules`에 더미 모듈을 미리 채워 넣어(`langchain_community.llms.VertexAI`를 상속한 껍데기 클래스) 우회했다. 설치된 패키지 파일을 직접 고치지 않고 우리 스크립트 안에서만 우회했다 — 재현 가능하고 유지보수하기 쉽다.
- **ragas 0.4.x의 "새 API"(`ragas.metrics.collections`)는 우리 스택에 안 맞음 → "레거시 API"로 우회.** 최신 ragas는 `instructor` 라이브러리 기반의 구조화 출력 LLM(`llm_factory`)을 요구하는 새 메트릭 클래스를 밀고 있는데, 이건 OpenAI 클라이언트를 전제로 한 패턴이라 우리의 `ChatBedrockConverse` 기반 스택과 안 맞았다. 대신 `ragas.metrics`가 `__getattr__`로 하위 호환 제공하는 레거시 싱글턴(`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`)을 썼다 — deprecated 경고는 뜨지만 `LangchainLLMWrapper`와 정상 동작한다.
- **동시성을 낮췄지만 문제가 안 풀림 → 원인 재규명.** "쓰로틀링이면 동시 요청을 줄이면 되지 않을까"라고 가정했으나 틀렸다. `max_workers=1`로도 거의 모든 호출이 즉시 실패해서, 이게 순간 부하(버스트) 문제가 아니라 하루 누적 총량 문제라는 걸 역으로 확인했다.
- **스크립트 버그로 이전 성공 데이터를 잃을 뻔함 → 발견 후 즉시 수정.** 지표를 하나씩 재시도하도록 스크립트를 고치면서, "이번 실행 결과로 해당 지표 열을 덮어쓰는" 로직을 짰는데, 이게 이전 실행에서 이미 성공했던 값까지 무조건 덮어써 버렸다. 실제로 1차 실행에서 성공했던 `context_recall` 4건이 2차 실행(전부 실패)으로 지워질 뻔했다. 대화 기록에 남아있던 원래 값으로 수동 복구하고, 스크립트는 "이번에 성공한 값만 갱신하고, 실패하면 이전 값을 보존"하도록 고쳤다. 이후 로직: `results.json`은 항상 "지금까지 확보한 것 중 최선"을 담는다.

## 5. 결론 및 다음 라운드 제안

- RAGAS 파이프라인 자체는 **끝까지 검증됐다** — 별도 1문항 테스트에서 `faithfulness: 0.9048`을 실제로 얻었고, 배선(에이전트 → 데이터셋 → judge LLM → 지표)에는 문제가 없다. 8건 배치가 못 끝난 건 순전히 계정 할당량 문제이지 구현 문제가 아니다.
- 이번 라운드에서 얻은 가장 중요한 인사이트는 숫자가 아니라 **"이 계정의 쓰로틀링은 요청 속도 문제가 아니라 하루 총량 문제"**라는 확인이다. 앞으로 이 계정으로 대량 평가를 계획할 때는 "천천히 보내기"가 아니라 "적게 보내기" 또는 "날짜를 나눠 보내기" 전략을 써야 한다.
- **다음 라운드(3차) 제안**:
  1. round1_report.md 6절의 미착수 과제 4개를 우선 처리 (HITL 2턴 실측 · 자동 채점 스크립트 · CSV `expected_tools` 재검토 · 실용성 후속 확인) — 3라운드 넘게 이월된 상태이므로 최우선
  2. 할당량이 회복된 시점에 `python evaluation/run_ragas.py` 재실행으로 `faithfulness`/`answer_relevancy` 8건, `context_recall`/`context_precision` 나머지 표본 확보 (캐시된 답변을 재사용하므로 에이전트 재호출 없이 지표만 재계산됨)
  3. RAGAS 표본을 지식 문서당 1건(8건)에서 늘릴지 검토 — 지금은 문서-질문이 1:1이라 검색 난이도가 낮다. 여러 문서에 걸친 질문(예: "짠 음식 좋아하는데 나트륨이랑 오메가3 둘 다 신경 써야 해?")을 추가하면 `context_precision`이 더 의미 있어진다
  4. `test_queries.csv`의 RAG 케이스(id=3, 8, 9, 12)와 RAGAS의 8문항 세트가 서로 다른 질문 세트로 운영되고 있다 — 장기적으로는 하나로 합치거나, 최소한 서로 참조하도록 정리할 필요가 있다
