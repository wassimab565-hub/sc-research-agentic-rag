import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from sc_research_agentic_rag.embeddings import Embedder, EmbeddingError
from sc_research_agentic_rag.evaluation.dataset import (
    DatasetError,
    load_chunk_corpus,
    load_dataset,
    validate_dataset,
)
from sc_research_agentic_rag.evaluation.metrics import QuestionResult, aggregate
from sc_research_agentic_rag.evaluation.models import EvalQuestion
from sc_research_agentic_rag.evaluation.runner import RetrieveFn, run_evaluation
from sc_research_agentic_rag.retrieval import Retriever
from sc_research_agentic_rag.storage import VectorStore, VectorStoreError

DEFAULT_CHUNKS_DIR = Path("data/chunks")
DEFAULT_RESULTS_DIR = Path("data/evaluation/results")


def build_retrieve_fn() -> tuple[RetrieveFn, str, Callable[[], None]]:
    embedder = Embedder.from_settings()
    store = VectorStore.from_settings(embedder.model_name)
    retriever = Retriever(embedder, store)
    return (lambda question, k: retriever.retrieve(question, top_k=k)), embedder.model_name, store.close


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on question datasets.")
    parser.add_argument("--dataset", action="append", type=Path, required=True, help="JSONL file (repeatable)")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--label", default="dense", help="name of the retrieval setup, saved with the results")
    parser.add_argument("--validate", action="store_true", help="only check the datasets against the chunks")
    parser.add_argument("--show-misses", type=int, default=0, metavar="N")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args(argv)
    if any(k < 1 for k in args.k):
        parser.error("--k values must be >= 1")
    ks = sorted(set(args.k))

    try:
        datasets = {path.stem: load_dataset(path) for path in args.dataset}
        corpus = load_chunk_corpus(args.chunks_dir)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not corpus:
        print(f"error: no chunks found in {args.chunks_dir}", file=sys.stderr)
        return 1

    failed = False
    for name, questions in datasets.items():
        issues = validate_dataset(questions, corpus)
        for issue in issues:
            print(f"{issue.level}: [{name}] {issue.question_id}: {issue.message}", file=sys.stderr)
        failed = failed or any(issue.level == "error" for issue in issues)
    if failed:
        return 1
    if args.validate:
        print(f"datasets are valid: {', '.join(f'{n} ({len(q)} questions)' for n, q in datasets.items())}")
        return 0

    try:
        retrieve, model_name, close = build_retrieve_fn()
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        all_results = {name: run_evaluation(q, retrieve, max(ks)) for name, q in datasets.items()}
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        close()

    summary = {name: aggregate(results, ks) for name, results in all_results.items()}
    for name, results in all_results.items():
        _print_metrics(name, len(results), summary[name], ks)
        if args.show_misses:
            _print_misses(datasets[name], results, args.show_misses)

    path = _save(args, ks, model_name, datasets, all_results, summary)
    print(f"\nresults saved to {path}")
    return 0


def _print_metrics(name: str, count: int, metrics: dict[str, float], ks: list[int]) -> None:
    print(f"\n{name} ({count} questions)")
    for k in ks:
        print(f"  Recall@{k:<3} {metrics[f'recall@{k}']:.3f}")
    print(f"  MRR        {metrics['mrr']:.3f}")
    print(f"  nDCG@{ks[-1]:<4} {metrics[f'ndcg@{ks[-1]}']:.3f}")


def _print_misses(questions: Sequence[EvalQuestion], results: Sequence[QuestionResult], limit: int) -> None:
    by_id = {question.id: question for question in questions}
    worst = sorted(results, key=lambda r: (r.first_rank is not None, -(r.first_rank or 0)))[:limit]
    print("  worst questions:")
    for result in worst:
        question = by_id[result.question_id]
        found = f"rank {result.first_rank}" if result.first_rank else "not found"
        wanted = question.evidence[0]
        print(f"    - {question.id} ({found}): {question.question}")
        print(f"      expected: {wanted.paper_uid} page {wanted.page}; got: {', '.join(result.retrieved_ids[:3])}")


def _save(args, ks, model_name, datasets, all_results, summary) -> Path:
    now = datetime.now(UTC)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{now:%Y%m%dT%H%M%SZ}_{args.label}.json"
    payload = {
        "created_at": now.isoformat(),
        "label": args.label,
        "embedding_model": model_name,
        "k": ks,
        "datasets": {
            name: {
                "questions": len(datasets[name]),
                "metrics": summary[name],
                "results": [
                    {"id": r.question_id, "first_rank": r.first_rank, "retrieved": r.retrieved_ids}
                    for r in results
                ],
            }
            for name, results in all_results.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path