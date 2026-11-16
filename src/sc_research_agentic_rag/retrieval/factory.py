from sc_research_agentic_rag.retrieval.bm25 import Bm25Retriever
from sc_research_agentic_rag.retrieval.hybrid import HybridRetriever
from sc_research_agentic_rag.retrieval.retriever import QueryEmbedder, Retriever
from sc_research_agentic_rag.storage.vector_store import VectorStore

MODES = ("dense", "hybrid")


def build_retriever(mode: str, embedder: QueryEmbedder, store: VectorStore) -> Retriever | HybridRetriever:
    dense = Retriever(embedder, store)
    if mode == "dense":
        return dense
    if mode == "hybrid":
        return HybridRetriever(dense, Bm25Retriever(store))
    raise ValueError(f"unknown retrieval mode '{mode}' (expected one of: {', '.join(MODES)})")
