# Milestones

| Phase | Status | Notes |
|-------|--------|-------|
| 0 - Project Foundation | Done | src layout, packaging, settings, smoke tests |
| 1 - Scientific Data Ingestion | Done | Paper model, arXiv + OpenAlex clients, retries, rate limiting, deduplication, search script |
| 2 - Document Processing | Done | PDF download, pypdf extraction (layout fallback), cleaning, chunking with page provenance, processing script |
| 3 - Basic RAG | Done | BGE-M3 embeddings, Qdrant (local), indexer, retriever with filters, context assembly, Ollama (qwen3:8b), cited answers, score threshold |
| 4 - Advanced Retrieval | In progress | 4.1 evaluation done (manual 12 + synthetic 150 questions; dense baseline: manual R@1 0.667 / MRR 0.792, synthetic R@1 0.527 / MRR 0.648). Next: 4.2 BM25, 4.3 hybrid RRF, 4.4 reranking (optional), 4.5 integration |