# retriever.py - RAG 파이프라인
#
# data/knowledge/*.md 문서를 읽어 BM25 검색 인덱스를 만들고,
# 질의와 가장 관련 있는 문서를 찾아 반환한다.
#
# 한국어는 조사·어미(이/가/을/를, -합니다/-해요 등) 때문에 어절 단위로 그냥 매칭하면
# "철분이"와 "철분을"이 다른 단어 취급된다. kiwipiepy로 형태소 분석한 뒤 명사·동사·부사
# 같은 내용어만 남기고, 그 토큰들을 rank_bm25(BM25Okapi)로 채점한다
# (requirements.txt의 "RAG (2일차~)" 스택: kiwipiepy + rank-bm25).
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"

# frontmatter(---로 감싼 YAML)와 본문을 분리하는 정규식
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)

# 형태소 분석기는 초기화 비용이 있으므로 모듈 로드 시 한 번만 만든다.
_kiwi = Kiwi()

# 검색에 남길 품사: 체언(N*), 용언(V*), 어근(XR), 외래어(SL), 숫자(SN), 일반부사(MAG).
# 조사(J*)·어미(E*)·구두점(S* 중 SL/SN 제외) 같은 기능어는 버려서 의미어 위주로 매칭한다.
_CONTENT_TAG_PREFIXES = ("N", "V", "XR", "SL", "SN")


def _tokenize(text: str) -> list[str]:
    """형태소 분석 후 명사·동사·부사 등 내용어만 남긴 토큰 리스트를 반환합니다."""
    return [
        tok.form
        for tok in _kiwi.tokenize(text)
        if tok.tag.startswith(_CONTENT_TAG_PREFIXES) or tok.tag == "MAG"
    ]


@dataclass
class KnowledgeChunk:
    """지식 문서 한 건. 현재는 문서 전체(360~460자)를 청크 하나로 취급한다."""

    doc_id: str
    title: str
    source: str
    text: str


def _load_documents() -> list[KnowledgeChunk]:
    """data/knowledge/*.md 를 모두 읽어 frontmatter와 본문을 분리합니다.

    frontmatter가 없거나 필수 필드(id/title/source)가 빠진 파일은 건너뛴다.
    """
    chunks: list[KnowledgeChunk] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(raw)
        if not match:
            continue
        meta = yaml.safe_load(match.group(1)) or {}
        if not {"id", "title", "source"} <= meta.keys():
            continue
        body = match.group(2).strip()
        chunks.append(
            KnowledgeChunk(doc_id=meta["id"], title=meta["title"], source=meta["source"], text=body)
        )
    return chunks


class NutritionRetriever:
    """영양 지식 문서 검색기.

    인스턴스 생성 시 한 번만 문서를 형태소 분석해 BM25 인덱스를 만든 뒤 재사용한다.
    """

    def __init__(self) -> None:
        self._chunks = _load_documents()
        self._tokenized_corpus = [_tokenize(f"{c.title}\n{c.text}") for c in self._chunks]
        # tokenizer 인자를 넘기면 rank_bm25가 내부적으로 멀티프로세싱 Pool로 코퍼스를
        # 토큰화하는데, Windows(spawn 방식)에서 Kiwi 인스턴스가 안전하게 pickle되지
        # 않을 수 있다. 그래서 코퍼스는 미리 우리가 직접 토큰화해 넘긴다.
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    def search(self, query: str, top_k: int = 2, min_score: float = 2.0) -> list[dict]:
        """질의와 가장 관련 있는 문서를 top_k개 반환합니다.

        BM25 점수가 min_score 미만인 문서는 "관련 없음"으로 보고 제외한다.
        결과가 비어 있으면 지식베이스에 근거가 없다는 뜻이므로, 호출하는 쪽(에이전트)이
        "관련 문서를 찾지 못했다"고 답해야 한다 (환각 방지, SERVICE.md 정책 3).

        min_score=2.0은 실제 8개 문서로 실험해 정한 값이다: "아이 키 크는 영양제
        추천해줘"처럼 근거 문서가 없어야 하는 질문의 최고 점수가 1.9대였고, 실제로
        관련 있는 질문들의 최저 점수는 2.39 이상이었다. 문서가 늘어나면 이 값도
        다시 보정해야 한다.
        """
        if self._bm25 is None:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked[:top_k]:
            if scores[i] < min_score:
                continue
            chunk = self._chunks[i]
            results.append(
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "text": chunk.text,
                    "score": round(float(scores[i]), 4),
                }
            )
        return results


# 모듈 로드 시 한 번만 인덱스를 구축해 재사용한다 (매 요청마다 다시 만들지 않는다).
_retriever = NutritionRetriever()


def search_nutrition_guidelines(query: str, top_k: int = 2) -> list[dict]:
    """이 모듈의 공개 진입점. tools.py가 이 함수를 감싸 LangChain 도구로 노출한다."""
    return _retriever.search(query, top_k=top_k)
