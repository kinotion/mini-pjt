# -*- coding: utf-8 -*-
"""run_ragas.py - RAG 파이프라인(search_nutrition_guidelines)에 RAGAS 4개 지표를 계산한다.

data/knowledge/*.md 8개 문서 각각에 대응하는 질문 1개씩, 총 8건을 실제 에이전트로 돌려
answer/contexts를 얻은 뒤, ragas의 레거시 지표(faithfulness, answer_relevancy,
context_precision, context_recall)를 계산한다.

judge LLM은 agent.py가 쓰는 Bedrock Claude를, 임베딩은 Bedrock Titan Embed v2를 그대로
재사용한다 - 외부 서비스(OpenAI 등)를 새로 쓰지 않는다.

이 계정의 Bedrock 일일 토큰 한도가 낮아서(이 프로젝트 전체에서 반복적으로 겪은 문제),
지표 4개를 한 번에 병렬로 돌리면 ThrottlingException으로 대부분 실패한다. 그래서:
  - 에이전트 호출(질문 8건) 결과는 ragas_results.json에 캐시해두고, 이미 있으면 재사용한다
    (재실행할 때마다 에이전트를 다시 부르지 않아 할당량을 아낀다).
  - 지표는 한 번에 하나씩만, RunConfig(max_workers=1)로 완전 순차 실행한다.
  - 한 지표가 실패해도 그 전까지 계산된 지표는 파일에 즉시 저장된다(부분 결과 보존).

실행: (프로젝트 루트에서) .venv\\Scripts\\python.exe evaluation\\run_ragas.py
"""
from __future__ import annotations

import json
import sys
import types
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ragas 0.4.3이 이미 사라진 langchain_community.chat_models.vertexai 를 무조건 import하는
# 문제가 있어 실제로 쓰지 않는 더미 심볼로 우회한다 (langchain-community 0.4.2 기준).
from langchain_community.llms import VertexAI as _VertexAI_LLM  # noqa: E402


class _ChatVertexAIStub(_VertexAI_LLM):
    """ragas의 stale import 경로 호환용 더미 - 실제로는 전혀 쓰이지 않는다."""


_stub_mod = types.ModuleType("langchain_community.chat_models.vertexai")
_stub_mod.ChatVertexAI = _ChatVertexAIStub
sys.modules["langchain_community.chat_models.vertexai"] = _stub_mod

warnings.filterwarnings("ignore", category=DeprecationWarning)

import agent  # noqa: E402
from datasets import Dataset  # noqa: E402
from langchain_aws import BedrockEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms.base import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

# data/knowledge/*.md 8건에 1:1 대응하는 질문 + ground truth(문서 내용을 그대로 요약한 것).
CASES = [
    {
        "question": "이유식은 몇 개월부터 시작해?",
        "ground_truth": (
            "생후 4~6개월 무렵 목을 가누고 음식에 관심을 보이는 신호가 나타나면 이유식을 "
            "시작할 수 있다. 초기(4~6개월)는 미음, 중기(7~9개월)는 죽, 후기(10~12개월)는 "
            "진밥, 완료기(12개월 이후)는 일반식으로 진행한다."
        ),
    },
    {
        "question": "알레르기 유발식품은 언제부터, 어떻게 도입해야 해?",
        "ground_truth": (
            "달걀, 우유, 밀, 대두, 견과류, 생선, 갑각류 같은 알레르기 유발식품도 이유식 "
            "초기부터 소량씩 시도하며 관찰하는 것이 권장된다. 한 번에 한 가지 재료만, "
            "오전 시간대에 소량 제공하고 2~3일간 이상 반응을 관찰하며, 이상 반응 시 즉시 "
            "중단하고 소아청소년과 진료를 받는다."
        ),
    },
    {
        "question": "아이 철분 섭취 기준이 어떻게 돼?",
        "ground_truth": (
            "2020 한국인 영양소 섭취기준에 따른 철 권장섭취량은 1~2세 6mg/일, 3~5세 7mg/일이다. "
            "동물성 식품(헴철)은 흡수율이 높고, 식물성 식품(비헴철)은 비타민C와 함께 먹으면 "
            "흡수율이 높아진다."
        ),
    },
    {
        "question": "칼슘은 얼마나 먹여야 해?",
        "ground_truth": (
            "2020 한국인 영양소 섭취기준에서 1~2세 칼슘 충분섭취량은 500mg/일 수준이며 3~5세는 "
            "이보다 다소 높다(정확한 수치는 원문 확인 필요). 급원 식품은 유제품, 두부, 뼈째 먹는 "
            "생선, 녹황색 채소다."
        ),
    },
    {
        "question": "오메가3는 어떻게 섭취시켜야 해?",
        "ground_truth": (
            "오메가3(알파-리놀렌산, EPA, DHA)는 두뇌와 시력 발달에 필요한 필수 지방산으로, "
            "고등어·연어 등 등푸른 생선, 들기름, 호두 같은 견과류가 급원 식품이다. 알레르기가 "
            "없다면 주 1~2회 생선 섭취로 자연스럽게 보충하는 게 실용적이다."
        ),
    },
    {
        "question": "아이가 편식이 심한데 어떻게 해야 해?",
        "ground_truth": (
            "편식은 유아기에 흔한 발달 과정으로, 새로운 음식은 8~10회 이상 반복 노출하고 억지로 "
            "먹이거나 벌주지 않는다. 좋아하는 재료와 함께 조리하거나 아이가 조리 과정에 참여하게 "
            "하면 거부감이 줄고, 보호자가 골고루 먹는 모습을 보여주는 것도 효과적이다."
        ),
    },
    {
        "question": "간식은 하루에 몇 번, 어떻게 줘야 해?",
        "ground_truth": (
            "간식은 하루 1~2회, 식사 시간과 최소 1.5~2시간 간격을 두고 제공한다. 과자·사탕 같은 "
            "가공식품보다 과일·유제품·감자·고구마 같은 자연식품 위주로 구성하고, 정해진 시간과 "
            "장소에서 먹는 습관을 들인다."
        ),
    },
    {
        "question": "아이가 짠 음식을 자꾸 찾는데 나트륨 섭취 어떻게 줄여?",
        "ground_truth": (
            "조리 시 소금·간장 사용을 최소화해 어른 기준보다 훨씬 싱겁게 하고, 당류도 과일·곡류의 "
            "자연당 위주로 섭취하며 주스·탄산음료·과자의 첨가당은 제한한다. 국물 요리는 국물보다 "
            "건더기 위주로 먹이고 시판 제품은 영양성분표를 확인하는 습관을 들인다."
        ),
    },
]

OUT_PATH = Path(__file__).resolve().parent / "ragas_results.json"

METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


def _load_cached_rows() -> list[dict] | None:
    """이전 실행에서 저장해둔 답변/근거문서를 재사용한다(에이전트 재호출로 할당량 낭비 방지)."""
    if not OUT_PATH.exists():
        return None
    data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    rows = data.get("per_question")
    if rows and len(rows) == len(CASES):
        return rows
    return None


def _build_rows() -> list[dict]:
    """실제 에이전트로 8건을 돌려 answer/contexts를 얻는다."""
    rows = []
    print(f"[에이전트 호출] {len(CASES)}건 실행 중...")
    for i, case in enumerate(CASES):
        resp = agent.handle_query(case["question"], thread_id=f"ragas-{i}")
        contexts = [c["text"] for c in resp["contexts"]] or [""]
        rows.append(
            {
                "user_input": case["question"],
                "response": resp["answer"],
                "retrieved_contexts": contexts,
                "reference": case["ground_truth"],
            }
        )
        print(f"  [{i+1}/{len(CASES)}] {case['question'][:30]}... (근거 문서 {len(contexts)}건)")
    return rows


def main() -> None:
    rows = _load_cached_rows()
    if rows is not None:
        print(f"[캐시 재사용] {OUT_PATH.name}에 저장된 답변/근거문서 {len(rows)}건을 그대로 씁니다 "
              "(에이전트 재호출 없음).")
    else:
        rows = _build_rows()

    dataset = Dataset.from_dict(
        {
            "user_input": [r["user_input"] for r in rows],
            "response": [r["response"] for r in rows],
            "retrieved_contexts": [r["retrieved_contexts"] for r in rows],
            "reference": [r["reference"] for r in rows],
        }
    )

    ragas_llm = LangchainLLMWrapper(agent.llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=agent._REGION)
    )
    # 하루 토큰 한도가 낮은 계정이라 병렬 호출을 최소화한다: 완전 순차(1개씩),
    # 재시도도 짧게(어차피 '일일 한도' 문제는 몇 초 기다린다고 안 풀린다).
    run_config = RunConfig(max_workers=1, max_retries=1, max_wait=5)

    aggregate: dict[str, float] = {}
    for i, name in enumerate(METRICS):
        metric = METRICS[name]
        print(f"\n[{i+1}/{len(METRICS)}] {name} 계산 중 (순차 실행)...")
        try:
            result = evaluate(dataset, metrics=[metric], llm=ragas_llm, embeddings=ragas_embeddings,
                               run_config=run_config)
            per_q = result.to_pandas()[name].tolist()
            n_ok = sum(1 for v in per_q if v == v)
            for row, v in zip(rows, per_q):
                if v == v:  # NaN이 아니면(이번 실행에서 성공) 갱신
                    row[name] = round(float(v), 4)
                else:
                    row.setdefault(name, None)  # 이번엔 실패 - 이전 실행의 성공값이 있으면 보존
            # 집계는 이번 실행 결과가 아니라, 지금까지 누적된 per-row 값 전체로 다시 계산한다
            # (이전 실행에서 성공한 행이 이번 실행에서 실패해도 사라지지 않도록).
            known = [row[name] for row in rows if row.get(name) is not None]
            aggregate[name] = round(sum(known) / len(known), 4) if known else None
            print(f"  -> {name}: {aggregate[name]} (이번 실행 성공 {n_ok}/{len(rows)}건, "
                  f"누적 확보 {len(known)}/{len(rows)}건)")
        except Exception as e:
            print(f"  -> {name} 전체 실패: {type(e).__name__}: {e}")
            for row in rows:
                row.setdefault(name, None)
            known = [row[name] for row in rows if row.get(name) is not None]
            aggregate[name] = round(sum(known) / len(known), 4) if known else None

        # 지표 하나 끝날 때마다 즉시 저장 - 다음 지표가 실패해도 여기까지는 보존된다.
        out = {"n": len(CASES), "aggregate": aggregate, "per_question": rows}
        OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n=== 최종 결과 ===")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")
    print(f"\n저장 위치: {OUT_PATH}")


if __name__ == "__main__":
    main()
