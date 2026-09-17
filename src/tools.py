# tools.py - 도메인 도구
#
# 키즈밀플래너 에이전트가 사용하는 레시피 검색, 알레르기 검증, 장기 기억(자녀 정보·냉장고 재료)
# 조회/갱신, 식단표 생성·저장, 영양 지식 검색(RAG는 retriever.py에 위임) 도구를 정의한다.
#
# 중요: data/allergy_profile.json(알레르기 프로필)은 이 모듈에서 절대 쓰지 않는다(읽기 전용).
# 이유는 SERVICE.md 정책 8 참고 - 대화 오인식으로 알레르기 항목이 사라지는 사고를 막기 위함이다.
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from retriever import search_nutrition_guidelines as _search_nutrition_guidelines

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RECIPES_PATH = DATA_DIR / "recipes.json"
ALLERGEN_MAP_PATH = DATA_DIR / "allergen_map.json"
ALLERGY_PROFILE_PATH = DATA_DIR / "allergy_profile.json"
MEMORY_STORE_PATH = DATA_DIR / "memory_store.json"

# 식단표 저장 위치. data/는 "사용한 문서와 데이터"용이라 생성된 결과물을 넣지 않고,
# CLAUDE.md의 제출 폴더 구조(src/data/evaluation)와 분리된 별도 런타임 산출물 폴더를 쓴다.
OUTPUT_DIR = DATA_DIR.parent / "output"

DISCLAIMER = "이 식단은 참고용이며, 정확한 영양 상담은 소아과 전문의나 영양사와 상의하세요."


# ---------------------------------------------------------------------------
# 내부 헬퍼 (도구가 아니라 도구를 구현하기 위한 보조 함수)
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    """JSON 데이터 파일을 읽어 dict로 반환합니다."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    """dict를 JSON 데이터 파일로 저장합니다."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _now_iso() -> str:
    """change_log에 남길 현재 시각을 ISO 형식 문자열로 반환합니다."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _resolve_term(term: str, allergen_map: dict) -> list[str]:
    """알레르기 프로필의 term(사용자 표현)을 성분 목록으로 해석합니다.

    해석 순서(SERVICE.md 7.4): component_groups 키 -> components.aliases.
    also_avoid_ingredients는 재료를 직접 지정하는 경로라 여기서는 다루지 않는다.
    """
    groups = {k: v for k, v in allergen_map["component_groups"].items() if not k.startswith("_")}
    if term in groups:
        return groups[term]
    for comp_name, comp in allergen_map["components"].items():
        if term in comp["aliases"]:
            return [comp_name]
    return []


def _ingredient_display(ingredient_id: str, allergen_map: dict) -> str:
    """재료 ID를 사람이 읽을 표시명으로 변환합니다. 없으면 ID를 그대로 반환합니다."""
    spec = allergen_map["ingredients"].get(ingredient_id)
    return spec["display"] if spec else ingredient_id


def _excluded_ingredient_ids(child_name: str | None = None) -> set[str]:
    """allergy_profile.json을 재료 구성 사전으로 펼쳐, 제외해야 할 ingredient_id 집합을 계산합니다.

    child_name을 지정하지 않으면 v1 기준 유일한 자녀(첫 번째 항목)를 사용한다.
    해석되지 않는 term이 있으면 조용히 넘어가지 않고 예외를 던진다(fail-closed, SERVICE.md 7.1-(6)).
    """
    allergen_map = _load_json(ALLERGEN_MAP_PATH)
    profile = _load_json(ALLERGY_PROFILE_PATH)
    ing = allergen_map["ingredients"]

    children = profile["children"]
    child = children[0] if child_name is None else next(
        (c for c in children if c["name"] == child_name), children[0]
    )

    excluded: set[str] = set()
    for allergy in child["allergies"]:
        components = _resolve_term(allergy["term"], allergen_map)
        also_avoid = allergy.get("also_avoid_ingredients", [])
        if not components and not also_avoid:
            raise ValueError(
                f"알레르기 프로필의 '{allergy['term']}'을(를) 해석하지 못했습니다. "
                "data/allergen_map.json의 components/component_groups를 확인하세요."
            )
        for iid, spec in ing.items():
            if set(spec["components"]) & set(components):
                excluded.add(iid)
        excluded.update(also_avoid)
    return excluded


# ---------------------------------------------------------------------------
# 장기 기억 (자녀 정보 · 냉장고 재료)
# ---------------------------------------------------------------------------
@tool
def get_household_memory() -> str:
    """장기 기억에 저장된 자녀 정보(이름·나이·선호)와 냉장고 재료 목록을 조회합니다.

    알레르기 정보는 여기 포함되지 않습니다 - 알레르기가 뭔지는 get_allergy_profile로,
    특정 메뉴가 알레르기와 충돌하는지는 check_allergen으로 확인하세요.
    식단을 추천하거나 보유 재료를 언급하기 전에는 먼저 이 도구로 현재 상태를 확인하세요.
    """
    memory = _load_json(MEMORY_STORE_PATH)
    payload = {"children": memory["children"], "ingredients": memory["ingredients"]}
    return json.dumps(payload, ensure_ascii=False)


@tool
def get_allergy_profile() -> str:
    """아이의 알레르기 프로필(무엇에 알레르기가 있는지, 심각도, 메모)을 조회합니다.

    "우리 아이 알레르기가 뭐였지?", "알레르기 프로필 좀 보여줘"처럼 사용자가 프로필
    내용 자체를 직접 물어볼 때 사용하세요. 이 도구는 읽기 전용입니다 - 알레르기
    프로필을 바꾸는 도구는 이 시스템에 존재하지 않습니다(정책상 의도된 제약이며,
    변경하려면 보호자가 data/allergy_profile.json 파일을 직접 수정해야 합니다).
    특정 레시피가 실제로 안전한지 판정할 때는 이 도구가 아니라 check_allergen을 쓰세요.
    """
    profile = _load_json(ALLERGY_PROFILE_PATH)
    children = [
        {
            "name": c["name"],
            "allergies": [
                {"term": a["term"], "severity": a["severity"], "note": a.get("note", "")}
                for a in c["allergies"]
            ],
        }
        for c in profile["children"]
    ]
    return json.dumps({"children": children}, ensure_ascii=False)


@tool
def update_household_memory(
    action: Literal["update_child", "add_or_update_ingredient", "remove_ingredient"],
    child_name: str | None = None,
    age_months: int | None = None,
    dislikes: list[str] | None = None,
    ingredient_id: str | None = None,
    quantity: str | None = None,
    expiry_date: str | None = None,
) -> str:
    """사용자가 대화 중 명시적으로 알려준 내용으로만 장기 기억을 갱신합니다.

    action별 사용법:
      - "update_child": child_name으로 대상을 지정하고, age_months/dislikes 중
        사용자가 실제로 알려준 값만 채웁니다. 나머지는 None으로 두면 바뀌지 않습니다.
      - "add_or_update_ingredient": ingredient_id는 필수, quantity/expiry_date로
        수량·유통기한을 추가하거나 갱신합니다. 예: "우유 샀어" -> milk 추가.
      - "remove_ingredient": ingredient_id에 해당하는 재료를 재고에서 제거합니다.
        예: "두부 다 썼어" -> tofu 제거.

    사용자가 "샀어/생겼어/몇 개월 됐어"처럼 명시적으로 말했을 때만 호출하세요.
    추측으로 먼저 호출하지 않습니다.

    경고: 이 도구는 알레르기 정보를 절대 변경할 수 없습니다(그런 필드 자체가 없습니다).
    아이의 알레르기가 나아졌다는 이야기가 나와도 이 도구를 호출하지 말고,
    data/allergy_profile.json 파일을 보호자가 직접 수정해야 한다고 안내하세요.
    """
    memory = _load_json(MEMORY_STORE_PATH)
    detail = ""

    if action == "update_child":
        children = memory["children"]
        target = children[0] if child_name is None else next(
            (c for c in children if c["name"] == child_name), children[0]
        )
        changes = []
        if age_months is not None:
            target["age_months"] = age_months
            changes.append(f"나이={age_months}개월")
        if dislikes is not None:
            target["dislikes"] = dislikes
            changes.append(f"선호(dislikes)={dislikes}")
        target["updated_at"] = _now_iso()
        detail = f"{target['name']} " + ", ".join(changes) if changes else f"{target['name']} 변경 없음"

    elif action == "add_or_update_ingredient":
        if not ingredient_id:
            return "ingredient_id가 필요합니다."
        items = memory["ingredients"]
        existing = next((i for i in items if i["ingredient_id"] == ingredient_id), None)
        if existing is None:
            allergen_map = _load_json(ALLERGEN_MAP_PATH)
            items.append(
                {
                    "ingredient_id": ingredient_id,
                    "display": _ingredient_display(ingredient_id, allergen_map),
                    "quantity": quantity or "",
                    "expiry_date": expiry_date,
                    "updated_at": _now_iso(),
                }
            )
            detail = f"{ingredient_id} 추가 (수량={quantity})"
        else:
            if quantity is not None:
                existing["quantity"] = quantity
            if expiry_date is not None:
                existing["expiry_date"] = expiry_date
            existing["updated_at"] = _now_iso()
            detail = f"{ingredient_id} 갱신 (수량={quantity}, 유통기한={expiry_date})"

    elif action == "remove_ingredient":
        if not ingredient_id:
            return "ingredient_id가 필요합니다."
        before = len(memory["ingredients"])
        memory["ingredients"] = [i for i in memory["ingredients"] if i["ingredient_id"] != ingredient_id]
        removed = before != len(memory["ingredients"])
        detail = f"{ingredient_id} 제거" if removed else f"{ingredient_id}는 원래 재고에 없었음"

    memory.setdefault("change_log", []).append(
        {"at": _now_iso(), "action": action, "target": ingredient_id or child_name, "detail": detail}
    )
    memory["updated_at"] = _now_iso()
    _save_json(MEMORY_STORE_PATH, memory)
    return f"장기 기억을 갱신했습니다: {detail}"


# ---------------------------------------------------------------------------
# 레시피 검색 · 알레르기 검증
# ---------------------------------------------------------------------------
@tool
def search_recipe(
    age_months: int,
    meal_type: Literal["breakfast", "lunch", "dinner", "snack"] | None = None,
    nutrition_tag: str | None = None,
    prefer_ingredient_ids: list[str] | None = None,
    exclude_ingredient_ids: list[str] | None = None,
    limit: int = 5,
) -> str:
    """나이(개월)·끼니 종류·영양소 포커스로 레시피 후보를 구조화 검색합니다.

    - meal_type: breakfast/lunch/dinner/snack 중 하나. 생략하면 모든 끼니에서 찾습니다.
    - nutrition_tag: '철분'처럼 레시피의 nutrition_tags 값. 나이만으로 후보가 없을 때
      이 조건부터 완화(생략)해서 다시 검색해 보세요.
    - prefer_ingredient_ids: 냉장고 등 우선 활용하고 싶은 재료 ID. 이 재료를 쓰는 레시피가
      앞쪽에 정렬됩니다 (get_household_memory로 조회한 보유 재료를 넘기세요).
    - exclude_ingredient_ids: 이번 검색에서 아예 배제할 재료 ID. 특정 재료 때문에
      check_allergen에서 후보가 전부 막혔다면, 그 재료를 여기 담아 다시 검색하세요
      (재검색 - ReAct 재계획).
    - 이 도구는 나이·끼니 조건으로 1차 후보만 추립니다. 알레르기 안전성은 이 도구가 아니라
      반드시 check_allergen으로 최종 확인해야 합니다.
    """
    recipes = _load_json(RECIPES_PATH)["recipes"]
    prefer = set(prefer_ingredient_ids or [])
    exclude = set(exclude_ingredient_ids or [])

    def matches(r: dict) -> bool:
        if not (r["min_age_months"] <= age_months and (r["max_age_months"] is None or age_months <= r["max_age_months"])):
            return False
        if meal_type and meal_type not in r["meal_types"]:
            return False
        if nutrition_tag and nutrition_tag not in r["nutrition_tags"]:
            return False
        used_ids = {i["ingredient_id"] for i in r["ingredients"]}
        if exclude & used_ids:
            return False
        return True

    def sort_key(r: dict) -> tuple:
        used_ids = {i["ingredient_id"] for i in r["ingredients"]}
        prefer_count = len(prefer & used_ids)
        return (-prefer_count, r["cook_minutes"])

    candidates = sorted((r for r in recipes if matches(r)), key=sort_key)[:limit]
    result = [
        {
            "id": r["id"],
            "name": r["name"],
            "meal_types": r["meal_types"],
            "ingredients": [i["ingredient_id"] for i in r["ingredients"]],
            "nutrition_tags": r["nutrition_tags"],
            "uses_preferred_ingredient": bool(prefer & {i["ingredient_id"] for i in r["ingredients"]}),
        }
        for r in candidates
    ]
    return json.dumps(result, ensure_ascii=False)


@tool
def check_allergen(recipe_ids: list[str]) -> str:
    """레시피 ID 목록을 받아 각 레시피가 아이의 알레르기 프로필과 충돌하는지 확인합니다.

    레시피 재료를 구성 성분까지 펼쳐서 판정하므로 간장 속 밀, 마요네즈 속 달걀 같은
    숨은 알레르겐도 잡아냅니다. optional 재료가 걸리면 메뉴를 통째로 버리지 않고
    "이 재료를 빼면 안전하다(safe_with_substitution)"고 알려줍니다.
    optional이 아닌 필수 재료가 걸리면 그 레시피는 unsafe입니다 - 절대 추천하지 마세요.
    식단을 최종 확정하기 전에는 반드시 이 도구로 안전 여부를 확인해야 합니다.
    """
    allergen_map = _load_json(ALLERGEN_MAP_PATH)
    recipes = {r["id"]: r for r in _load_json(RECIPES_PATH)["recipes"]}
    excluded = _excluded_ingredient_ids()

    results = []
    for rid in recipe_ids:
        recipe = recipes.get(rid)
        if recipe is None:
            results.append({"recipe_id": rid, "status": "not_found"})
            continue

        blocking = [i for i in recipe["ingredients"] if not i["optional"] and i["ingredient_id"] in excluded]
        removable = [i for i in recipe["ingredients"] if i["optional"] and i["ingredient_id"] in excluded]

        if blocking:
            names = [_ingredient_display(i["ingredient_id"], allergen_map) for i in blocking]
            results.append(
                {"recipe_id": rid, "name": recipe["name"], "status": "unsafe", "blocking_ingredients": names}
            )
        elif removable:
            names = [_ingredient_display(i["ingredient_id"], allergen_map) for i in removable]
            results.append(
                {"recipe_id": rid, "name": recipe["name"], "status": "safe_with_substitution", "remove": names}
            )
        else:
            results.append({"recipe_id": rid, "name": recipe["name"], "status": "safe"})

    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 영양 지식 검색 (RAG 위임)
# ---------------------------------------------------------------------------
@tool
def search_nutrition_guidelines(query: str) -> str:
    """영양·이유식·알레르기 도입·편식 등 일반적인 육아 지식 질문에 대한 근거 문서를 검색합니다.

    반드시 이 도구가 찾아준 문서 내용에 근거해서만 답하세요 - 지어내면 안 됩니다(환각 방지).
    검색 결과가 비어 있으면 지식베이스에 근거가 없다는 뜻이므로,
    "관련 문서를 찾지 못했다"고 솔직하게 답하세요.
    """
    results = _search_nutrition_guidelines(query, top_k=2)
    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 식단표 생성 · 저장 (저장은 위험 도구 - HITL 승인 필요)
# ---------------------------------------------------------------------------
# v1은 한 번에 최대 3일치(9끼)까지만 지원한다 (SERVICE.md 0장). 시스템 프롬프트로
# 모델이 4일 이상 요청을 미리 걸러내도록 안내하지만, 모델이 그래도 4일치를 다 검색해
# 이 도구까지 넘겨버리는 경우를 대비해 여기서도 한 번 더 막는다(안전망).
MAX_PLAN_DAYS = 3


@tool
def generate_meal_plan_table(plan: list[dict]) -> str:
    """확정된 끼니별 메뉴 목록을 마크다운 표로 정리합니다.

    plan 각 원소는 다음 키를 가진 dict입니다:
      day(int), meal_type(str), recipe_name(str), fridge_used(bool), note(str, 선택)
    이 표는 아직 파일로 저장되지 않은 상태입니다 - 저장하려면 save_meal_plan_to_file을
    별도로 호출해야 하며, 그 호출은 보호자 승인을 거칩니다.

    v1은 최대 3일치까지만 지원합니다. plan에 4일 이상이 섞여 있으면 표를 만들지 않고
    안내 메시지만 반환합니다 - 그런 경우 3일 단위로 나눠서 다시 요청하세요.
    """
    days = {item.get("day") for item in plan}
    if len(days) > MAX_PLAN_DAYS:
        return (
            f"v1은 한 번에 최대 {MAX_PLAN_DAYS}일치 식단표만 지원합니다 "
            f"(지금 요청은 {len(days)}일치입니다). 3일 단위로 나눠서 다시 요청해 주세요."
        )

    header = "| 일자 | 끼니 | 메뉴 | 냉장고 활용 | 비고 |\n|---|---|---|---|---|\n"
    meal_label = {"breakfast": "아침", "lunch": "점심", "dinner": "저녁", "snack": "간식"}
    rows = []
    for item in plan:
        rows.append(
            "| {day}일차 | {meal} | {name} | {fridge} | {note} |".format(
                day=item.get("day", ""),
                meal=meal_label.get(item.get("meal_type", ""), item.get("meal_type", "")),
                name=item.get("recipe_name", ""),
                fridge="O" if item.get("fridge_used") else "",
                note=item.get("note", ""),
            )
        )
    table = header + "\n".join(rows)
    return table


@tool
def save_meal_plan_to_file(markdown_table: str, filename: str = "meal_plan.md") -> str:
    """생성된 식단표(마크다운)를 로컬 파일로 저장합니다.

    위험 도구입니다 - 보호자의 명시적 승인(HITL) 없이는 실행되지 않습니다.
    승인 절차 자체는 agent.py의 HumanInTheLoopMiddleware가 처리합니다.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_text(markdown_table, encoding="utf-8")
    return f"식단표를 {path}에 저장했습니다."


# ---------------------------------------------------------------------------
# 선호 제약 해제 (알레르기 아님 - 위험 도구, HITL 승인 필요)
# ---------------------------------------------------------------------------
@tool
def override_preference_constraint(ingredient_or_term: str) -> str:
    """편식·선호 제약(알레르기 아님)을 이번 요청에 한해 해제합니다.

    예: 아이가 평소 싫어하는 재료(dislikes)를 그래도 넣어달라는 요청에 사용합니다.
    위험 도구입니다 - 보호자의 명시적 승인(HITL) 없이는 실행되지 않습니다.

    이 도구는 알레르기 재료에는 절대 쓸 수 없습니다. 코드 레벨에서 알레르기 프로필과
    겹치는지 다시 한번 확인하며, 겹치면 승인 여부와 무관하게 거부합니다
    (정책 1·2 - 알레르기 우회 수단은 애초에 제공하지 않는다).
    """
    allergen_map = _load_json(ALLERGEN_MAP_PATH)
    excluded = _excluded_ingredient_ids()

    # ingredient_or_term이 재료 ID/표시명/별칭 중 무엇으로 오든 재료 ID로 최대한 정규화한다.
    target_id = None
    for iid, spec in allergen_map["ingredients"].items():
        if ingredient_or_term in (iid, spec["display"], *spec.get("aliases", [])):
            target_id = iid
            break

    if target_id and target_id in excluded:
        return (
            f"'{ingredient_or_term}'은(는) 알레르기 프로필과 관련된 재료라 해제할 수 없습니다. "
            "알레르기는 어떤 승인으로도 우회되지 않습니다. 편식(dislikes) 재료만 해제할 수 있습니다."
        )

    return f"'{ingredient_or_term}' 제약을 이번 식단 생성에 한해 해제했습니다."
