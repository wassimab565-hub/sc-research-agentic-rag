import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sc_research_agentic_rag.evaluation.dataset import DatasetError, load_chunk_corpus, question_to_line
from sc_research_agentic_rag.evaluation.synthetic import generate_questions
from sc_research_agentic_rag.llm import LLMError, OllamaClient, OllamaSettings

DEFAULT_CHUNKS_DIR = Path("data/chunks")
DEFAULT_OUTPUT = Path("data/evaluation/synthetic.jsonl")


def build_client() -> OllamaClient:
    return OllamaClient(OllamaSettings())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation questions from chunks.")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be >= 1")

    try:
        corpus = load_chunk_corpus(args.chunks_dir)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not corpus:
        print(f"error: no chunks found in {args.chunks_dir}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with args.output.open("w", encoding="utf-8") as handle:
            for question in generate_questions(corpus, build_client(), args.count, args.seed):
                handle.write(question_to_line(question) + "\n")
                handle.flush()
                written += 1
                if written % 10 == 0:
                    print(f"  {written}/{args.count} questions", flush=True)
    except LLMError as exc:
        print(f"error: {exc} ({written} questions were saved)", file=sys.stderr)
        return 1

    print(f"{written} questions saved to {args.output}")
    return 0 if written else 1