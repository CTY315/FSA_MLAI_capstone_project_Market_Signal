"""RAG pipeline: index S&P 500 headlines into ChromaDB and query via LangChain + Claude.

Embeddings use chromadb's built-in ONNX runtime (all-MiniLM-L6-v2) to avoid the
torch / numpy version-incompatibility that the sentence_transformers Python package
would introduce.  The model and its vectors are identical to the ML pipeline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.embeddings import Embeddings as _LCEmbeddings

from src.utils import DATA_END, DATA_RAW, PROJECT_ROOT

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_DIR = PROJECT_ROOT / "data" / "chromadb"
COLLECTION_NAME = "sp500_headlines"
TOP_K = 15
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = (
    "You are a financial market analyst assistant. "
    "Use the retrieved headlines as context to answer the user's question. "
    "Focus on specific events, companies, and market moves mentioned in the headlines — "
    "do not give generic summaries. Quote or reference specific headlines where relevant. "
    "If the headlines don't contain relevant information, say so explicitly."
)
_PROMPT_TEMPLATE = (
    f"{SYSTEM_PROMPT}\n\n"
    "Retrieved headlines:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# Headline quality filters applied at index time
_GENERIC_PATTERNS = re.compile(
    r"^stock market news for\b",
    re.IGNORECASE,
)
_MIN_TITLE_LEN = 20  # characters after stripping

# Month name → number mapping for date parsing
_MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


# ── Date parsing ──────────────────────────────────────────────────────────────
def _parse_date_filter(question: str) -> dict[str, Any] | None:
    """Extract a ChromaDB metadata $and/$gte/$lte filter from the question text.

    Handles:
    - "March 2020" / "March of 2020"  → full month range
    - "2020"                           → full year range
    - "Q1 2020" / "first quarter 2020" → quarter range
    - "early/mid/late 2020"            → partial-year range

    Returns a ChromaDB ``where`` dict, or None if no date was detected.
    Dates outside 2008–2023 are clamped silently.
    """
    q = question.lower()

    # ── Quarter patterns ───────────────────────────────────────────────────
    quarter_words = {
        "first quarter": 1, "q1": 1,
        "second quarter": 2, "q2": 2,
        "third quarter": 3, "q3": 3,
        "fourth quarter": 4, "q4": 4,
    }
    _quarter_month_start = {1: 1, 2: 4, 3: 7, 4: 10}
    _quarter_month_end   = {1: 3, 2: 6, 3: 9, 4: 12}

    for label, qnum in quarter_words.items():
        m = re.search(rf"{re.escape(label)}\s+(?:of\s+)?(\d{{4}})", q)
        if m:
            year = int(m.group(1))
            ms, me = _quarter_month_start[qnum], _quarter_month_end[qnum]
            import calendar
            last_day = calendar.monthrange(year, me)[1]
            return _build_where(f"{year}-{ms:02d}-01", f"{year}-{me:02d}-{last_day:02d}")

    # Also handle "Q1 2020" with uppercase Q in the original
    m = re.search(r"\bq([1-4])\s+(\d{4})\b", question, re.IGNORECASE)
    if m:
        qnum = int(m.group(1))
        year = int(m.group(2))
        ms, me = _quarter_month_start[qnum], _quarter_month_end[qnum]
        import calendar
        last_day = calendar.monthrange(year, me)[1]
        return _build_where(f"{year}-{ms:02d}-01", f"{year}-{me:02d}-{last_day:02d}")

    # ── Partial-year vague qualifiers ──────────────────────────────────────
    vague = {
        r"\bearly\b": (1, 4),
        r"\bmid(?:\-| )?year\b|\bmiddle\b": (4, 8),
        r"\blate\b": (9, 12),
    }
    for pattern, (ms, me) in vague.items():
        m = re.search(rf"{pattern}.{{0,20}}(\d{{4}})", q)
        if m:
            year = int(m.group(1))
            import calendar
            last_day = calendar.monthrange(year, me)[1]
            return _build_where(f"{year}-{ms:02d}-01", f"{year}-{me:02d}-{last_day:02d}")

    # ── Month + year ───────────────────────────────────────────────────────
    month_pattern = "|".join(re.escape(k) for k in sorted(_MONTH_MAP, key=len, reverse=True))
    m = re.search(rf"({month_pattern})[^a-z0-9]{{0,10}}(\d{{4}})", q)
    if not m:
        # Also handle "2020 March" order
        m = re.search(rf"(\d{{4}})[^a-z0-9]{{0,10}}({month_pattern})\b", q)
        if m:
            year, month_name = int(m.group(1)), m.group(2)
        else:
            month_name, year = None, None
    else:
        month_name, year = m.group(1), int(m.group(2))

    if month_name and year:
        month_num = _MONTH_MAP[month_name]
        import calendar
        last_day = calendar.monthrange(int(year), month_num)[1]
        return _build_where(
            f"{year}-{month_num:02d}-01",
            f"{year}-{month_num:02d}-{last_day:02d}",
        )

    # ── Year only ──────────────────────────────────────────────────────────
    m = re.search(r"\b(200[89]|20[12]\d)\b", q)
    if m:
        year = int(m.group(1))
        return _build_where(f"{year}-01-01", f"{year}-12-31")

    return None


def _date_to_int(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to integer YYYYMMDD for ChromaDB numeric comparison."""
    return int(date_str.replace("-", ""))


def _build_where(start: str, end: str) -> dict[str, Any]:
    """Build a ChromaDB $and filter for a date range (inclusive, YYYY-MM-DD strings).

    Dates are stored and compared as integers (YYYYMMDD) because ChromaDB's
    $gte/$lte operators require numeric operands.
    """
    return {"$and": [{"date_int": {"$gte": _date_to_int(start)}}, {"date_int": {"$lte": _date_to_int(end)}}]}


# ── Embedding wrapper (ONNX, no torch) ───────────────────────────────────────
class _ONNXEmbeddings(_LCEmbeddings):
    """LangChain-compatible wrapper around chromadb's DefaultEmbeddingFunction.

    Uses all-MiniLM-L6-v2 via onnxruntime — the same model as the ML pipeline
    but without the torch / sentence_transformers stack.  The ONNX weights are
    cached at ~/.cache/chroma/ after the first call.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        # chromadb 1.5+ requires numpy arrays or native Python floats,
        # not a Python list of np.float32 scalars.
        return [np.asarray(v, dtype=np.float32) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        import numpy as np

        return np.asarray(self._fn([text])[0], dtype=np.float32)


# ── Internal helpers ──────────────────────────────────────────────────────────
def _get_embeddings() -> _ONNXEmbeddings:
    return _ONNXEmbeddings()


def _get_vectorstore(embeddings: _ONNXEmbeddings | None = None):
    """Load the existing ChromaDB persistent vectorstore via LangChain."""
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma  # type: ignore[no-redef]

    if embeddings is None:
        embeddings = _get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def _is_informative(title: str) -> bool:
    """Return True if the headline is worth indexing.

    Skips generic placeholders ('Stock Market News for …') and very short
    titles that carry no semantic signal for retrieval.
    """
    stripped = title.strip()
    if len(stripped) < _MIN_TITLE_LEN:
        return False
    if _GENERIC_PATTERNS.match(stripped):
        return False
    return True


# ── Public API ────────────────────────────────────────────────────────────────
def is_indexed() -> bool:
    """Return True if the ChromaDB collection exists and contains documents."""
    if not CHROMA_DIR.exists():
        return False
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False


def index_headlines(force: bool = False) -> int:
    """Index informative headlines from sp500_headlines.csv into ChromaDB.

    Generic placeholders ('Stock Market News for …') and titles shorter than
    20 characters are skipped to improve retrieval quality.

    Args:
        force: Drop and recreate the collection if it already exists.

    Returns:
        Number of documents indexed.
    """
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma  # type: ignore[no-redef]
    from langchain_core.documents import Document

    # Load, date-filter, and quality-filter headlines
    csv_path = DATA_RAW / "sp500_headlines.csv"
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= DATA_END].dropna(subset=["title"]).reset_index(drop=True)

    before = len(df)
    df = df[df["title"].apply(_is_informative)].reset_index(drop=True)
    skipped = before - len(df)
    logger.info(
        "Filtered out %d generic/short headlines (%d remaining).", skipped, len(df)
    )

    logger.info("Indexing %d headlines into %s …", len(df), CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if force and CHROMA_DIR.exists():
        import shutil

        shutil.rmtree(CHROMA_DIR)
        logger.info("Wiped existing ChromaDB directory at %s.", CHROMA_DIR)

    embeddings = _get_embeddings()

    docs = [
        Document(
            page_content=row["title"],
            metadata={
                "date": row["date"].strftime("%Y-%m-%d"),
                "date_int": int(row["date"].strftime("%Y%m%d")),
            },
        )
        for _, row in df.iterrows()
    ]

    # Index in batches (chromadb per-call limit ~41 000)
    BATCH = 5_000
    vectorstore: Any = None
    for start in range(0, len(docs), BATCH):
        batch = docs[start : start + BATCH]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_DIR),
            )
        else:
            vectorstore.add_documents(batch)
        logger.info("  … %d / %d", min(start + BATCH, len(docs)), len(docs))

    logger.info("Indexing complete.")
    return len(docs)


class _RAGChain:
    """Minimal RAG chain using langchain_core + langchain_anthropic only.

    Avoids RetrievalQA / langchain_classic, which transitively import
    sentence_transformers and crash on the torch/numpy version mismatch
    present in this venv.

    Date-aware retrieval: if the question mentions a time period, a ChromaDB
    metadata filter is applied before semantic search to restrict candidates to
    that window.
    """

    def __init__(self, vectorstore: Any, llm: Any, prompt_template: str) -> None:
        self._vectorstore = vectorstore
        self._llm = llm
        self._prompt_template = prompt_template

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        from langchain_core.documents import Document

        question = inputs["query"]

        # Date-aware retrieval
        where_filter = _parse_date_filter(question)
        if where_filter:
            logger.debug("Applying date filter: %s", where_filter)
            docs: list[Document] = self._vectorstore.similarity_search(
                question, k=TOP_K, filter=where_filter
            )
            # Fall back to unfiltered if date filter returns nothing
            if not docs:
                logger.debug("Date filter returned 0 results — falling back to global search.")
                docs = self._vectorstore.similarity_search(question, k=TOP_K)
        else:
            docs = self._vectorstore.similarity_search(question, k=TOP_K)

        # Sort by date so context and source list are in chronological order
        docs.sort(key=lambda d: d.metadata.get("date_int", 0))

        context = "\n".join(
            f"[{d.metadata.get('date', '')}] {d.page_content}" for d in docs
        )
        full_prompt = self._prompt_template.format(
            context=context, question=question
        )
        response = self._llm.invoke(full_prompt)
        return {
            "result": response.content,
            "source_documents": docs,
        }


def build_rag_chain(api_key: str) -> _RAGChain:
    """Build a RAG chain backed by ChromaDB and Claude Haiku.

    Args:
        api_key: Anthropic API key.

    Returns:
        _RAGChain instance with a date-aware .invoke() method.
    """
    from langchain_anthropic import ChatAnthropic

    embeddings = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings)

    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        anthropic_api_key=api_key,
        max_tokens=1024,
    )

    return _RAGChain(vectorstore, llm, _PROMPT_TEMPLATE)


def query_rag(chain: Any, question: str) -> tuple[str, list[dict[str, str]]]:
    """Query the RAG chain and return the answer with source headline metadata.

    Args:
        chain: _RAGChain from :func:`build_rag_chain`.
        question: User's natural-language question.

    Returns:
        ``(answer, sources)`` where *sources* is a list of
        ``{"date": "YYYY-MM-DD", "headline": "..."}`` dicts.
    """
    result = chain.invoke({"query": question})
    answer: str = result.get("result", "")
    sources = [
        {
            "date": doc.metadata.get("date", "unknown"),
            "headline": doc.page_content,
        }
        for doc in result.get("source_documents", [])
    ]
    return answer, sources
