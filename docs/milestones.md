# Milestones

| Phase | Status | Notes |
|-------|--------|-------|
| 0 - Project Foundation | Done | src layout, packaging, settings, smoke tests |
| 1 - Scientific Data Ingestion | Done | Paper model, arXiv + OpenAlex clients, retries, rate limiting, deduplication, search script |
| 2 - Document Processing | Done | PDF download, pypdf extraction (layout fallback), cleaning, chunking with page provenance, processing script |
| 3 - Basic RAG | Done | BGE-M3 embeddings, Qdrant (local), indexer, retriever with filters, context assembly, Ollama (qwen3:8b), cited answers, score threshold |
| 4 - Advanced Retrieval | Next | Evaluation set, BM25, hybrid fusion, reranking |