# api.py - 웹 프론트 서빙 + POST /query API 서버
#
# SERVICE.md 4장 규약대로 POST /query 로 question을 받아 answer/contexts/trace를 돌려준다.
# HITL 승인 대기 여부는 서버가 따로 플래그를 들고 있지 않고, 매 요청마다
# graph.get_state(...).interrupts 로 LangGraph 체크포인터 상태를 직접 확인한다
# (agent.py의 InMemorySaver가 살아있는 프로세스 안에서는 항상 정확하다).
from __future__ import annotations

from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from agent import graph, handle_query

STATIC_DIR = Path(__file__).resolve().parent / "static"

# v1은 단일 가정(household_id="family_001")만 다루므로 스레드도 고정값 하나를 쓴다
# (SERVICE.md 4장). 즉 이 서버를 여는 모든 브라우저 탭이 같은 대화를 공유한다 -
# 여러 사용자가 동시에 쓰는 배포라면 세션별 thread_id 발급이 필요하다.
THREAD_ID = "default"

_YES = {"y", "yes", "예", "네", "승인", "approve"}
_NO = {"n", "no", "아니오", "아니요", "거절", "reject"}


def _pending_action_requests() -> list[dict]:
    """현재 스레드가 승인 대기 상태라면 대기 중인 작업 목록을, 아니면 빈 리스트를 반환합니다."""
    config = {"configurable": {"thread_id": THREAD_ID}}
    snapshot = graph.get_state(config)
    if not snapshot.interrupts:
        return []
    return snapshot.interrupts[0].value["action_requests"]


async def query_endpoint(request: Request) -> JSONResponse:
    """POST /query. {"question": "..."} 을 받아 answer/contexts/trace 세 키를 돌려줍니다.

    직전 응답이 pending_approval 상태였다면, 이번 question이 y/n(과 그 변형)인지 먼저
    확인해서 approve/reject로 변환해 그래프를 재개한다. 그 판단은 여기서만 하고,
    agent.handle_query에는 판단이 끝난 resume_decision만 넘긴다.
    """
    body = await request.json()
    question = str(body.get("question", "")).strip()

    pending = _pending_action_requests()
    if pending:
        normalized = question.lower()
        if normalized in _YES:
            response = handle_query(question, thread_id=THREAD_ID, resume_decision="approve")
        elif normalized in _NO:
            response = handle_query(
                question, thread_id=THREAD_ID, resume_decision="reject", reject_message=question
            )
        else:
            # 승인 대기 중에 y/n이 아닌 말이 들어오면 그래프를 건드리지 않고 다시 안내만 한다.
            lines = "\n".join(f"- {a['name']}({a['args']})" for a in pending)
            response = {
                "answer": f"먼저 아래 작업을 승인('y')하거나 거절('n')해 주세요:\n{lines}",
                "contexts": [],
                "trace": {"status": "pending_approval", "pending": pending, "calls": []},
            }
    else:
        response = handle_query(question, thread_id=THREAD_ID)

    return JSONResponse(response)


async def index(request: Request) -> FileResponse:
    """웹 프론트 페이지(static/index.html)를 서빙합니다."""
    return FileResponse(STATIC_DIR / "index.html")


app = Starlette(
    routes=[
        Route("/", index),
        Route("/query", query_endpoint, methods=["POST"]),
    ],
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
