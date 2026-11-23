# src\sc_research_agentic_rag\agent\
import argparse
import os
import sys
from collections.abc import Callable, Sequence

from dotenv import load_dotenv

from sc_research_agentic_rag.agent.config import AgentSettings
from sc_research_agentic_rag.agent.pipeline import AgentAnswer, AgentPipeline
from sc_research_agentic_rag.agent.tools import AgentTools
from sc_research_agentic_rag.embeddings import Embedder, EmbeddingError
from sc_research_agentic_rag.graph import GraphError, GraphQueries, GraphStore
from sc_research_agentic_rag.ingestion.cli import build_sources
from sc_research_agentic_rag.llm import LLMError, OllamaClient
from sc_research_agentic_rag.rag.cli import _print_result
from sc_research_agentic_rag.rag.config import RagSettings
from sc_research_agentic_rag.retrieval import MODES, RetrievalSettings, build_retriever
from sc_research_agentic_rag.storage import SearchFilter, VectorStore, VectorStoreError

from sc_research_agentic_rag.verification import DEFAULT_THRESHOLD, build_report, build_support_embedder

def build_pipeline(
    max_steps: int, max_retrievals: int, top_k: int, score_threshold: float | None, mode: str
) -> tuple[AgentPipeline, Callable[[], None]]:
    embedder = Embedder.from_settings()
    store = VectorStore.from_settings(embedder.model_name)
    try:
        retriever = build_retriever(mode, embedder, store)
    except VectorStoreError:
        store.close()
        raise

    graph = _open_graph()
    load_dotenv()
    searchers = build_sources(["arxiv", "openalex"], os.environ.get("OPENALEX_API_KEY") or None)
    tools = AgentTools(
        retriever,
        GraphQueries(graph) if graph is not None else None,
        searchers,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    pipeline = AgentPipeline(OllamaClient(), tools, max_steps=max_steps, max_retrievals=max_retrievals)

    def close() -> None:
        store.close()
        if graph is not None:
            graph.close()

    return pipeline, close


def _open_graph() -> GraphStore | None:
    try:
        return GraphStore.from_settings()
    except GraphError as exc:
        print(f"warning: graph tool disabled ({exc})", file=sys.stderr)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the research agent (evidence retrieval, graph, paper search).")
    parser.add_argument("question")
    parser.add_argument("--max-steps", type=int, help="maximum tool calls (default: AGENT_MAX_STEPS)")
    parser.add_argument("--max-retrievals", type=int, help="maximum evidence searches (default: AGENT_MAX_RETRIEVALS)")
    parser.add_argument("--top-k", type=int, help="passages per evidence search (default: AGENT_TOP_K)")
    parser.add_argument("--mode", choices=MODES, help="retrieval mode (default: RETRIEVAL_MODE)")
    parser.add_argument("--paper", action="append", metavar="UID", help="restrict to a paper (repeatable)")
    parser.add_argument("--source", action="append", choices=["arxiv", "openalex", "upload"])
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--threshold", type=float, help="minimum similarity score (default: RAG_SCORE_THRESHOLD)")
    parser.add_argument("--show-trace", action="store_true", help="print the steps of the agent")
    parser.add_argument("--show-sources", action="store_true", help="print every retrieved passage")
    parser.add_argument("--verify", action="store_true", help="check the citations of the answer")
    parser.add_argument("--support-threshold", type=float, default=DEFAULT_THRESHOLD, help="minimum claim support score for --verify")    
    args = parser.parse_args(argv)

    settings = AgentSettings()
    max_steps = args.max_steps if args.max_steps is not None else settings.max_steps
    max_retrievals = args.max_retrievals if args.max_retrievals is not None else settings.max_retrievals
    top_k = args.top_k if args.top_k is not None else settings.top_k
    if not args.question.strip():
        parser.error("question must not be empty")
    if max_steps < 1:
        parser.error("--max-steps must be >= 1")
    if max_retrievals < 1:
        parser.error("--max-retrievals must be >= 1")
    if top_k < 1:
        parser.error("--top-k must be >= 1")
    if not 0 <= args.support_threshold <= 1:
        parser.error("--support-threshold must be between 0 and 1")
    try:
        filters = SearchFilter(
            paper_uids=args.paper, sources=args.source, year_min=args.year_min, year_max=args.year_max
        )
    except ValueError as exc:
        parser.error(str(exc))

    threshold = args.threshold if args.threshold is not None else RagSettings().score_threshold
    mode = args.mode or RetrievalSettings().mode
    close = None
    try:
        pipeline, close = build_pipeline(max_steps, max_retrievals, top_k, threshold, mode)
        result = pipeline.ask(args.question, filters)
    except (EmbeddingError, VectorStoreError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if close is not None:
            close()

    _print_result(result.as_rag_answer(), args.show_sources)
    if result.invalid_citations and not result.sources:
        numbers = ", ".join(f"[{n}]" for n in result.invalid_citations)
        print(f"Warning: the answer cites unknown sources: {numbers}")
    if args.verify:
        report = build_report(
            result.answer, result.sources, has_notes=_has_notes(result),
            threshold=args.support_threshold, embedder_factory=build_support_embedder,
        )
        print("\n" + report)
    if args.show_trace:
        _print_trace(result)
    return 0


def _print_trace(result: AgentAnswer) -> None:
    print(f"\nAgent steps (stopped: {result.stop_reason}):")
    if not result.trace:
        print("  none")
    for step in result.trace:
        print(f"  {step.number}. {step.decision.action} {step.decision.args} -> {step.summary}")

def _has_notes(result: AgentAnswer) -> bool:
    return any(step.found and step.decision.action != "retrieve_evidence" for step in result.trace)