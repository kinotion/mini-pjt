# -*- coding: utf-8 -*-
"""run_eval.py - test_queries.csv 20건을 실제 에이전트로 돌려 결과를 JSON으로 저장한다.

각 케이스를 독립된 스레드(thread_id=<prefix>-<id>)로 agent.handle_query()에 호출한다.
케이스 하나가 실패해도 나머지는 계속 진행되며, 도구 호출 순서(trace)와 최종 답변을 함께
저장한다. 판정(pass/fail)은 이 스크립트가 하지 않는다 - expected_traits/forbidden 같은
자연어 조건은 사람이 답변을 읽고 직접 판정해야 한다(round1_report.md/round2_report.md 참고).

실행: (프로젝트 루트에서) .venv\\Scripts\\python.exe evaluation\\run_eval.py [prefix]
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent  # noqa: E402

CSV_PATH = Path(__file__).resolve().parent / "test_queries.csv"


def main(prefix: str = "eval") -> None:
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    results = []

    for row in rows:
        rid = row["id"]
        thread_id = f"{prefix}-{rid}"
        t0 = time.time()
        try:
            resp = agent.handle_query(row["input"], thread_id=thread_id)
            elapsed = round(time.time() - t0, 1)
            results.append(
                {
                    "id": rid,
                    "category": row["category"],
                    "input": row["input"],
                    "answer": resp["answer"],
                    "contexts": resp["contexts"],
                    "trace_status": resp["trace"]["status"],
                    "tool_calls": [c["tool"] for c in resp["trace"]["calls"]],
                    "elapsed_s": elapsed,
                    "error": None,
                }
            )
            print(f"[OK]   id={rid} ({elapsed}s) tools={[c['tool'] for c in resp['trace']['calls']]}", flush=True)
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            results.append(
                {
                    "id": rid,
                    "category": row["category"],
                    "input": row["input"],
                    "answer": None,
                    "contexts": None,
                    "trace_status": None,
                    "tool_calls": None,
                    "elapsed_s": elapsed,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            print(f"[FAIL] id={rid} ({elapsed}s) {type(e).__name__}: {e}", flush=True)

    out_path = Path(__file__).resolve().parent / f"{prefix}_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval")
