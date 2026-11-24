# src\sc_research_agentic_rag\evaluation\
import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from sc_research_agentic_rag.embeddings import EmbeddingError
from sc_research_agentic_rag.evaluation.answer_runs import (
    AskFn,
    RunEvaluation,
    SavedAnswer,
    collect_answers,
    evaluate_run,
    load_answers,
    saved_answer,
    support_curve,
)
from sc_research_agentic_rag.evaluation.dataset import DatasetError, load_dataset
from sc_research_agentic_rag.evaluation.models import EvalQuestion
from sc_research_agentic_rag.llm import LLMError
from sc_research_agentic_rag.rag.config import RagSettings
from sc_research_agentic_rag.rag.context import DEFAULT_MAX_CONTEXT_CHARS
from sc_research_agentic_rag.retrieval import MODES, RetrievalSettings
from sc_research_agentic_rag.storage import VectorStoreError
from sc_research_agentic_rag.verification import DEFAULT_THRESHOLD, build_support_embedder

DEFAULT_DATASET = Path("data/evaluation/manual.jsonl")
DEFAULT_ANSWERS_DIR = Path("data/evaluation/results")
CURVE = (0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_TOP_K = 5


def build_asker(
    agent: bool, mode: str, top_k: int | None, max_steps: int | None, max_retrievals: int | None
) -> tuple[AskFn, Callable[[], None]]:
    """The function that answers one question (plain RAG or agent) and the function that closes its stores."""
    threshold = RagSettings().score_threshold
    if agent:
        from sc_research_agentic_rag.agent import cli as agent_cli  # needs langgraph: only loaded for --agent
        from sc_research_agentic_rag.agent.config import AgentSettings

        settings = AgentSettings()
        pipeline, close = agent_cli.build_pipeline(
            max_steps or settings.max_steps, max_retrievals or settings.max_retrievals,
            top_k or settings.top_k, threshold, mode,
        )
    else:
        from sc_research_agentic_rag.rag import cli as rag_cli

        pipeline, store = rag_cli.build_pipeline(top_k or DEFAULT_TOP_K, DEFAULT_MAX_CONTEXT_CHARS, threshold, None, mode)
        close = store.close
    return (lambda question: saved_answer(question.id, question.question, pipeline.ask(question.question))), close


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer the evaluation questions (RAG or agent) and verify the answers.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--agent", action="store_true", help="use the agent instead of plain RAG")
    parser.add_argument("--mode", choices=MODES, help="retrieval mode (default: RETRIEVAL_MODE)")
    parser.add_argument("--label", help="name of the run (default: rag-<mode> or agent-<mode>)")
    parser.add_argument("--limit", type=int, help="maximum number of NEW questions to answer")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-steps", type=int, help="agent only")
    parser.add_argument("--max-retrievals", type=int, help="agent only")
    parser.add_argument("--support-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--no-claims", action="store_true", help="skip the claim checks (no embedding model)")
    parser.add_argument("--show-claims", action="store_true", help="print every claim with its support score")
    parser.add_argument("--answers-dir", type=Path, default=DEFAULT_ANSWERS_DIR)
    args = parser.parse_args(argv)
    for name in ("limit", "top_k", "max_steps", "max_retrievals"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if not 0 <= args.support_threshold <= 1:
        parser.error("--support-threshold must be between 0 and 1")

    mode = args.mode or RetrievalSettings().mode
    label = args.label or f"{'agent' if args.agent else 'rag'}-{mode}"
    cache = args.answers_dir / f"answers_{label}.jsonl"
    try:
        questions = load_dataset(args.dataset)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        answers = _collect(args, questions, cache, mode)
    except (EmbeddingError, VectorStoreError, LLMError) as exc:
        print(f"error: {exc} (progress is saved: run the command again to resume)", file=sys.stderr)
        return 1
    answered = [q for q in questions if q.id in answers]
    if not answered:
        print("error: no answers to evaluate", file=sys.stderr)
        return 1

    try:
        result = evaluate_run(questions, answers, None if args.no_claims else build_support_embedder(), args.support_threshold)
    except EmbeddingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_run(label, len(answered), len(questions), result, args.show_claims)
    path = _save(args, label, mode, result, answers)
    print(f"\nresults saved to {path}")
    return 0


def _collect(
    args: argparse.Namespace, questions: Sequence[EvalQuestion], cache: Path, mode: str
) -> dict[str, SavedAnswer]:
    answers = load_answers(cache)
    if all(question.id in answers for question in questions):
        return answers  # nothing to ask: no store, no LLM
    ask, close = build_asker(args.agent, mode, args.top_k, args.max_steps, args.max_retrievals)
    try:
        return collect_answers(questions, ask, cache, args.limit, lambda done: print(f"  {done} answered", flush=True))
    finally:
        close()


def _print_run(label: str, answered: int, total: int, result: RunEvaluation, show_claims: bool) -> None:
    print(f"\n{label}: {answered} answers (of {total} questions)")
    print("  statuses: " + ", ".join(f"{name} {count}" for name, count in Counter(c.status for c in result.checks).items()))
    for name, value in result.metrics.items():
        print(f"  {name:<28} {value:.3f}")
    curve = support_curve([claim for items in result.claims.values() for claim in items], CURVE)
    if curve:
        print("  support >= t: " + " | ".join(f"{t:.1f}: {share:.2f}" for t, share in curve.items()))
    if show_claims:
        print("\nClaims:")
        for question_id, items in result.claims.items():
            for claim in items:
                score = f" {claim.support:.2f}" if claim.support is not None else ""
                print(f'  {question_id} [{claim.status}{score}] "{claim.text}"')


def _save(args: argparse.Namespace, label: str, mode: str, result: RunEvaluation, answers: dict[str, SavedAnswer]) -> Path:
    now = datetime.now(UTC)
    args.answers_dir.mkdir(parents=True, exist_ok=True)
    path = args.answers_dir / f"answers_{now:%Y%m%dT%H%M%SZ}_{label}.json"
    payload = {
        "created_at": now.isoformat(),
        "label": label,
        "mode": mode,
        "agent": args.agent,
        "dataset": str(args.dataset),
        "support_threshold": args.support_threshold,
        "metrics": result.metrics,
        "questions": [
            {
                **asdict(check),
                "stop_reason": answers[check.question_id].stop_reason,
                "claims": [asdict(claim) for claim in result.claims[check.question_id]],
            }
            for check in result.checks
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
