# src\sc_research_agentic_rag\api\
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from sc_research_agentic_rag.agent import AgentPipeline, AgentSettings, AgentTools
from sc_research_agentic_rag.api.schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    AuthorPaperOut,
    AuthorResponse,
    ClaimOut,
    ConceptPaperOut,
    ConceptResponse,
    NameCountOut,
    PaperOut,
    PassageOut,
    RelatedPaperOut,
    RelatedResponse,
    StepOut,
    VerificationOut,
)
from sc_research_agentic_rag.api.services import AppServices, build_services
from sc_research_agentic_rag.embeddings import EmbeddingError
from sc_research_agentic_rag.graph import GraphError, GraphQueries
from sc_research_agentic_rag.llm import LLMError
from sc_research_agentic_rag.rag import RagPipeline
from sc_research_agentic_rag.storage import VectorStoreError
from sc_research_agentic_rag.verification import check_claims, verify_citations

DEFAULT_TOP_K = 5
HEALTH_TIMEOUT = 5.0


def get_services(request: Request) -> AppServices:
    return request.app.state.services


def _error_handler(status: int):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    return handler


def create_app(services: AppServices | None = None) -> FastAPI:
    """`services` is given by tests; without it the services are built at startup (lifespan)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = not hasattr(app.state, "services")
        if owned:
            app.state.services = build_services()
        try:
            yield
        finally:
            if owned:
                app.state.services.close()

    app = FastAPI(title="SC-Research-Agentic-RAG", lifespan=lifespan)
    if services is not None:
        app.state.services = services

    for error in (LLMError, EmbeddingError, VectorStoreError, GraphError):
        app.add_exception_handler(error, _error_handler(503))
    app.add_exception_handler(ValueError, _error_handler(422))

    @app.get("/health")
    def health(services: AppServices = Depends(get_services)) -> dict[str, Any]:
        if services.lock.acquire(blocking=False):  # never wait behind a long question
            try:
                qdrant = _probe(lambda: {"chunks": services.store.count()})
            finally:
                services.lock.release()
        else:
            qdrant = {"ok": True, "busy": True}
        ollama = _probe(lambda: {"models": services.llm.list_models()})
        if services.graph is None:
            graph = {"ok": False, "error": "not configured"}
        else:
            graph = _probe(lambda: {"counts": services.graph.counts()})
        status = "ok" if qdrant["ok"] and ollama["ok"] else "degraded"
        return {"status": status, "default_mode": services.default_mode, "qdrant": qdrant, "ollama": ollama, "graph": graph}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, services: AppServices = Depends(get_services)) -> AskResponse:
        filters = body.filters.to_search_filter() if body.filters else None
        threshold = body.threshold if body.threshold is not None else services.score_threshold
        pipeline = RagPipeline(
            services.retriever(body.mode),
            services.llm,
            top_k=body.top_k or DEFAULT_TOP_K,
            score_threshold=threshold,
            max_per_paper=body.max_per_paper,
        )
        with services.lock:
            result = pipeline.ask(body.question, filters)
            verification = (
                _verify(services, result, False, body.support_threshold) if body.verify else None
            )
        return AskResponse(**_answer_fields(result, verification))

    @app.post("/agent/ask", response_model=AgentResponse)
    def agent_ask(body: AgentRequest, services: AppServices = Depends(get_services)) -> AgentResponse:
        settings = AgentSettings()
        filters = body.filters.to_search_filter() if body.filters else None
        threshold = body.threshold if body.threshold is not None else services.score_threshold
        tools = AgentTools(
            services.retriever(body.mode),
            GraphQueries(services.graph) if services.graph is not None else None,
            services.searchers,
            top_k=body.top_k or settings.top_k,
            score_threshold=threshold,
            list_size=settings.list_size,
        )
        pipeline = AgentPipeline(
            services.llm,
            tools,
            max_steps=body.max_steps or settings.max_steps,
            max_retrievals=body.max_retrievals or settings.max_retrievals,
        )
        with services.lock:
            result = pipeline.ask(body.question, filters)
            verification = (
                _verify(services, result, _has_notes(result), body.support_threshold) if body.verify else None
            )
        trace = [
            StepOut(
                number=step.number, action=step.decision.action, args=dict(step.decision.args),
                summary=step.summary, found=step.found,
            )
            for step in result.trace
        ]
        return AgentResponse(**_answer_fields(result, verification), stop_reason=result.stop_reason, trace=trace)

    @app.get("/papers", response_model=list[PaperOut])
    def papers(services: AppServices = Depends(get_services)) -> list[PaperOut]:
        with services.lock:
            return collect_papers(services.store.scroll_payloads())

    @app.get("/graph/concept", response_model=ConceptResponse)
    def graph_concept(
        name: str = Query(min_length=1),
        limit: int = Query(10, ge=1, le=100),
        services: AppServices = Depends(get_services),
    ) -> ConceptResponse:
        queries = _queries(services)
        found = queries.papers_for_concept(name, limit)
        similar = [] if found else queries.similar_concepts(name)
        return ConceptResponse(
            name=name,
            papers=[
                ConceptPaperOut(
                    uid=p.uid, title=p.title, year=p.year, authors=list(p.authors),
                    pages=list(p.pages), chunks=p.chunks,
                )
                for p in found
            ],
            similar_concepts=[NameCountOut(name=n, count=c) for n, c in similar],
        )

    @app.get("/graph/author", response_model=AuthorResponse)
    def graph_author(
        name: str = Query(min_length=1),
        limit: int = Query(10, ge=1, le=100),
        services: AppServices = Depends(get_services),
    ) -> AuthorResponse:
        queries = _queries(services)
        found = queries.papers_for_author(name, limit)
        concepts = queries.author_concepts(name) if found else []
        return AuthorResponse(
            name=name,
            matched_authors=sorted({author for p in found for author in p.matched_authors}),
            papers=[
                AuthorPaperOut(uid=p.uid, title=p.title, year=p.year, matched_authors=list(p.matched_authors))
                for p in found
            ],
            concepts=[NameCountOut(name=n, count=c) for n, c in concepts],
        )

    @app.get("/graph/related", response_model=RelatedResponse)
    def graph_related(
        uid: str = Query(min_length=1),
        limit: int = Query(10, ge=1, le=100),
        services: AppServices = Depends(get_services),
    ) -> RelatedResponse:
        found = _queries(services).related_papers(uid, limit)
        return RelatedResponse(
            uid=uid,
            papers=[
                RelatedPaperOut(uid=p.uid, title=p.title, year=p.year, score=p.score, shared=list(p.shared))
                for p in found
            ],
        )

    return app


def _probe(check) -> dict[str, Any]:
    try:
        return {"ok": True, **check()}
    except Exception as exc:  # noqa: BLE001 - a health check must report, never crash
        return {"ok": False, "error": str(exc)}


def _queries(services: AppServices) -> GraphQueries:
    if services.graph is None:
        raise GraphError("knowledge graph not available")
    return GraphQueries(services.graph)


def _has_notes(result: Any) -> bool:
    # Same rule as agent/cli._has_notes and evaluation/answer_runs.saved_answer (I-49: now three copies).
    return any(step.found and step.decision.action != "retrieve_evidence" for step in result.trace)


def _answer_fields(result: Any, verification: VerificationOut | None) -> dict[str, Any]:
    cited = {passage.number for passage in result.cited_sources}
    sources = [
        PassageOut(
            number=p.number, paper_uid=p.paper_uid, title=p.title, authors=list(p.authors), year=p.year,
            doi=p.doi, url=p.url, source=p.source, page=p.page, text=p.text, score=p.score,
            cited=p.number in cited,
        )
        for p in result.sources
    ]
    return {
        "question": result.question,
        "answer": result.answer,
        "sources": sources,
        "cited_numbers": sorted(cited),
        "invalid_citations": list(result.invalid_citations),
        "verification": verification,
    }


def _verify(services: AppServices, result: Any, has_notes: bool, threshold: float) -> VerificationOut:
    """Citation report (7.1) and claim checks (7.2) as data. Same logic as verification.report.build_report."""
    report = verify_citations(result.answer, [p.number for p in result.sources], has_notes)
    note = None
    try:
        checks = check_claims(result.answer, result.sources, services.support_embedder(), threshold)
    except EmbeddingError as exc:
        checks = check_claims(result.answer, result.sources, None, threshold)
        note = f"support score unavailable: {exc}"
    return VerificationOut(
        status=report.status,
        valid=list(report.valid),
        invalid=list(report.invalid),
        sentences=report.sentences,
        uncited_sentences=list(report.uncited_sentences),
        claims=[
            ClaimOut(
                text=c.text, citations=list(c.citations), status=c.status, support=c.support,
                missing_numbers=list(c.missing_numbers), found_in=list(c.found_in),
            )
            for c in checks
        ],
        note=note,
    )


def collect_papers(payloads: Iterable[dict[str, Any]]) -> list[PaperOut]:
    """One entry per paper, built from the chunk payloads of the vector store."""
    papers: dict[str, PaperOut] = {}
    for payload in payloads:
        uid = payload.get("paper_uid", "")
        known = papers.get(uid)
        if known is None:
            papers[uid] = PaperOut(
                paper_uid=uid, title=payload.get("title") or uid, authors=list(payload.get("authors") or []),
                year=payload.get("year"), source=payload.get("source", ""), doi=payload.get("doi"),
                url=payload.get("url"), chunks=1,
            )
        else:
            known.chunks += 1
    return sorted(papers.values(), key=lambda paper: paper.paper_uid)
