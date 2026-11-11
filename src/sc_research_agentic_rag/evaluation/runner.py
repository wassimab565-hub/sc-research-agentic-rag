from collections.abc import Callable, Sequence

from sc_research_agentic_rag.evaluation.metrics import QuestionResult, evaluate_question
from sc_research_agentic_rag.evaluation.models import EvalQuestion
from sc_research_agentic_rag.retrieval import RetrievedChunk

RetrieveFn = Callable[[str, int], list[RetrievedChunk]]


def run_evaluation(questions: Sequence[EvalQuestion], retrieve: RetrieveFn, max_k: int) -> list[QuestionResult]:
    if max_k < 1:
        raise ValueError("max_k must be >= 1")
    results = []
    for question in questions:
        retrieved = retrieve(question.question, max_k)
        results.append(evaluate_question(question, retrieved, [chunk.chunk_id for chunk in retrieved]))
    return results