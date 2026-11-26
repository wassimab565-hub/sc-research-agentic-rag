# src\sc_research_agentic_rag\api\
import os
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from sc_research_agentic_rag.embeddings import Embedder
from sc_research_agentic_rag.graph import GraphError, GraphStore
from sc_research_agentic_rag.ingestion.cli import build_sources
from sc_research_agentic_rag.ingestion.service import PaperSearcher
from sc_research_agentic_rag.llm import OllamaClient
from sc_research_agentic_rag.rag.config import RagSettings
from sc_research_agentic_rag.rag.pipeline import ChunkRetriever
from sc_research_agentic_rag.retrieval import MODES, RetrievalSettings, build_retriever
from sc_research_agentic_rag.storage import Indexer, VectorStore
from sc_research_agentic_rag.verification import build_support_embedder


@dataclass
class AppServices:
    """Everything the API shares between requests. Built once; `lock` serializes the heavy work
    (local Qdrant, the embedding model and the GPU are single-user)."""

    retrievers: Mapping[str, ChunkRetriever]
    llm: Any
    store: Any
    graph: Any | None = None
    searchers: Mapping[str, PaperSearcher] = field(default_factory=dict)
    default_mode: str = "dense"
    score_threshold: float | None = None
    support_embedder_factory: Callable[[], Any] = build_support_embedder
    lock: threading.Lock = field(default_factory=threading.Lock)
    _support_embedder: Any = field(default=None, init=False, repr=False)

    def retriever(self, mode: str | None) -> ChunkRetriever:
        return self.retrievers[mode or self.default_mode]

    def support_embedder(self) -> Any:
        """The claim-check embedder, created once (the model itself loads on first use)."""
        if self._support_embedder is None:
            self._support_embedder = self.support_embedder_factory()
        return self._support_embedder

    def close(self) -> None:
        self.store.close()
        if self.graph is not None:
            self.graph.close()


def _open_graph() -> GraphStore | None:
    try:
        return GraphStore.from_settings()
    except GraphError as exc:
        print(f"warning: graph disabled ({exc})", file=sys.stderr)
        return None


def build_services() -> AppServices:
    """Loads bge-m3, Qdrant (and the BM25 index), Ollama and Neo4j once, at API startup."""
    load_dotenv()
    embedder = Embedder.from_settings()
    store = VectorStore.from_settings(embedder.model_name)
    graph = None
    try:
        Indexer(embedder, store).prepare()  # creates an empty collection if needed and loads the model
        retrievers = {mode: build_retriever(mode, embedder, store) for mode in MODES}
        graph = _open_graph()
        searchers = build_sources(["arxiv", "openalex"], os.environ.get("OPENALEX_API_KEY") or None)
    except Exception:
        store.close()
        if graph is not None:
            graph.close()
        raise
    return AppServices(
        retrievers=retrievers,
        llm=OllamaClient(),
        store=store,
        graph=graph,
        searchers=searchers,
        default_mode=RetrievalSettings().mode,
        score_threshold=RagSettings().score_threshold,
    )
