# Milestones

| Phase | Status | Notes |
|-------|--------|-------|
| 0 - Project Foundation | Done | src layout, packaging, settings, smoke tests |
| 1 - Scientific Data Ingestion | Done | Paper model, arXiv + OpenAlex clients, retries, rate limiting, deduplication, search script |
| 2 - Document Processing | Done | PDF download, pypdf extraction (layout fallback), cleaning, chunking with page provenance, processing script |
| 3 - Basic RAG | Done | BGE-M3 embeddings, Qdrant (local), indexer, retriever with filters, context assembly, Ollama (qwen3:8b), cited answers, score threshold |
| 4 - Advanced Retrieval | In progress | 4.1 evaluation done. 4.2 done: internal BM25 + hybrid RRF (k=60, 50 candidates per side), `eval_retrieval.py --mode dense\|bm25\|hybrid`. Hybrid vs dense: manual R@1 0.750 / MRR 0.850 (dense 0.667 / 0.792), synthetic R@1 0.633 / MRR 0.739 (dense 0.527 / 0.648). Next: 4.3 integration (RETRIEVAL_MODE, resolve I-6), 4.4 reranking (optional) |