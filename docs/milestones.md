# Milestones

| Phase | Status | Notes |
|-------|--------|-------|
| 0 - Project Foundation | Done | src layout, packaging, settings, smoke tests |
| 1 - Scientific Data Ingestion | Done | Paper model, arXiv + OpenAlex clients, retries, rate limiting, deduplication, search script |
| 2 - Document Processing | Done | PDF download, pypdf extraction (layout fallback), cleaning, chunking with page provenance, processing script |
| 3 - Basic RAG | Done | BGE-M3 embeddings, Qdrant (local), indexer, retriever with filters, context assembly, Ollama (qwen3:8b), cited answers, score threshold |
| 4 - Advanced Retrieval | In progress | 4.1 evaluation, 4.2 BM25 + hybrid RRF (hybrid vs dense: manual R@1 0.750 / MRR 0.850 vs 0.667 / 0.792, synthetic R@1 0.633 / MRR 0.739 vs 0.527 / 0.648), 4.3 integration (RETRIEVAL_MODE, --mode in ask/search_chunks, dense-score gate for the threshold) done. Next: 4.4 reranking (optional, postponed) |
| 5 - GraphRAG | Done | 5.1 metadata graph (Neo4j in local Docker), 5.2 concept extraction (Ollama, 736 chunks, 2073 concepts), 5.3 GraphRetriever + eval modes `graph` / `hybrid+graph` (graph does NOT improve ranking: hybrid+graph R@1 0.487 vs hybrid 0.633 on synthetic), 5.4 relational queries without LLM (`graph_query.py --concept/--author/--related`) |
| 6 - Agentic RAG | Done | 6.1 agent tools + JSON decisions + CLI (`ask_agent.py`), 6.2 LangGraph (`agent/graph.py`: decide / act / finish, same behaviour, bounded by max steps and retrievals). Tools: hybrid retrieval, GraphQueries, read-only paper search |
| 7 - Verification and Evaluation | Done | 7.1 citation validation (`--verify`), 7.2 claim checks (exact number match + bge-m3 support score on CPU, misattributed citations), 7.3 answer evaluation runner (RAG vs agent on manual.jsonl). Own metrics, no Ragas, no LLM judge (7.4 not built). Support threshold 0.5 is provisional |
| 8 - API and UI | In progress | 8.1 + 8.2 FastAPI (health, ask, agent, verify, papers, graph) done. Next: 8.3 PDF upload, 8.4 + 8.5 Streamlit UI |