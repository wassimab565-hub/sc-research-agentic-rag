# src\sc_research_agentic_rag\agent\
from collections.abc import Mapping
from dataclasses import dataclass

from sc_research_agentic_rag.agent.decisions import Decision
from sc_research_agentic_rag.graph import GraphError, GraphQueries
from sc_research_agentic_rag.ingestion.service import PaperSearcher, collect_papers
from sc_research_agentic_rag.rag.pipeline import ChunkRetriever
from sc_research_agentic_rag.retrieval import RetrievedChunk
from sc_research_agentic_rag.storage import SearchFilter


@dataclass(frozen=True)
class ToolResult:
    """summary: one short line shown to the planner. chunks: citable evidence. note: text for the final answer."""

    summary: str
    found: int = 0
    chunks: tuple[RetrievedChunk, ...] = ()
    note: str = ""


class AgentTools:
    """The three tools of the agent. Graph and paper search are optional: a missing tool answers 'not available'."""

    def __init__(
        self,
        retriever: ChunkRetriever,
        graph: GraphQueries | None = None,
        searchers: Mapping[str, PaperSearcher] | None = None,
        top_k: int = 5,
        score_threshold: float | None = None,
        max_per_paper: int | None = None,
        list_size: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 1 <= list_size <= 100:
            raise ValueError("list_size must be between 1 and 100")
        self._retriever = retriever
        self._graph = graph
        self._searchers = dict(searchers or {})
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._max_per_paper = max_per_paper
        self._list_size = list_size

    def run(self, decision: Decision, filters: SearchFilter | None = None) -> ToolResult:
        args = decision.args
        if decision.action == "retrieve_evidence":
            return self.retrieve_evidence(args["query"], filters)
        if decision.action == "graph_lookup":
            return self.graph_lookup(args["kind"], args["value"])
        if decision.action == "search_papers":
            return self.search_papers(args["query"])
        raise ValueError(f"unknown action '{decision.action}'")

    def retrieve_evidence(self, query: str, filters: SearchFilter | None = None) -> ToolResult:
        chunks = self._retriever.retrieve(
            query,
            top_k=self._top_k,
            filters=filters,
            score_threshold=self._score_threshold,
            max_per_paper=self._max_per_paper,
        )
        if not chunks:
            return ToolResult("no passage found")
        listing = "; ".join(f"{_short(chunk.title)} p.{chunk.page}" for chunk in chunks[: self._list_size])
        summary = f'{len(chunks)} passages ({listing}). Best passage: "{_short(chunks[0].text, 150)}"'
        return ToolResult(summary, len(chunks), tuple(chunks))

    def graph_lookup(self, kind: str, value: str) -> ToolResult:
        if self._graph is None:
            return ToolResult("knowledge graph not available")
        try:
            if kind == "concept":
                return self._concept(value)
            if kind == "author":
                return self._author(value)
            if kind == "related":
                return self._related(value)
        except GraphError as exc:
            return ToolResult(f"knowledge graph unavailable: {_short(str(exc), 120)}")
        except ValueError as exc:
            return ToolResult(f"invalid request: {exc}")
        return ToolResult(f"unknown graph kind '{kind}'")

    def search_papers(self, query: str) -> ToolResult:
        if not self._searchers:
            return ToolResult("paper search not available")
        result = collect_papers(query, self._searchers, self._list_size)
        papers = result.papers[: self._list_size]
        errors = "; ".join(f"{name}: {message}" for name, message in result.errors.items())
        if not papers:
            return ToolResult("no paper found" + (f" ({_short(errors, 120)})" if errors else ""))
        lines = []
        for paper in papers:
            authors = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
            lines.append(f"- [{paper.uid}] {paper.year or 'n/a'} | {paper.title}" + (f" | {authors}" if authors else ""))
        note = f"Papers found by a literature search for '{query}' (not necessarily indexed):\n" + "\n".join(lines)
        summary = f"{len(papers)} papers: " + "; ".join(_short(paper.title) for paper in papers)
        return ToolResult(summary, len(papers), note=note)

    def _concept(self, name: str) -> ToolResult:
        papers = self._graph.papers_for_concept(name, self._list_size)
        if not papers:
            similar = self._graph.similar_concepts(name)
            hint = ("; similar concepts: " + ", ".join(concept for concept, _ in similar)) if similar else ""
            return ToolResult(f"no paper mentions '{name}'{hint}")
        lines = [
            f"- {_label(paper.title, paper.uid, paper.year)}: {paper.chunks} chunks, "
            f"pages {', '.join(str(page) for page in paper.pages)}"
            for paper in papers
        ]
        note = f"Papers that mention '{name}' (knowledge graph):\n" + "\n".join(lines)
        summary = f"{len(papers)} papers mention '{name}': " + ", ".join(paper.uid for paper in papers)
        return ToolResult(summary, len(papers), note=note)

    def _author(self, name: str) -> ToolResult:
        papers = self._graph.papers_for_author(name, self._list_size)
        if not papers:
            return ToolResult(f"no author matches '{name}'")
        matched = sorted({author for paper in papers for author in paper.matched_authors})
        lines = [f"- {_label(paper.title, paper.uid, paper.year)}" for paper in papers]
        note = f"Papers of authors matching '{name}' ({', '.join(matched)}):\n" + "\n".join(lines)
        concepts = self._graph.author_concepts(name, self._list_size)
        if concepts:
            note += "\nMain concepts: " + ", ".join(f"{concept} ({count})" for concept, count in concepts)
        return ToolResult(f"{len(papers)} papers by {', '.join(matched)}", len(papers), note=note)

    def _related(self, uid: str) -> ToolResult:
        papers = self._graph.related_papers(uid, self._list_size)
        if not papers:
            return ToolResult(f"no related paper for '{uid}' (unknown uid, or no extracted concepts)")
        lines = [
            f"- {_label(paper.title, paper.uid, paper.year)}, shared concepts: {', '.join(paper.shared)}"
            for paper in papers
        ]
        note = f"Papers related to {uid} (shared concepts, knowledge graph):\n" + "\n".join(lines)
        summary = f"{len(papers)} related papers: " + ", ".join(paper.uid for paper in papers)
        return ToolResult(summary, len(papers), note=note)


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _label(title: str | None, uid: str, year: int | None) -> str:
    return f"[{uid}] {_short(title or uid, 90)}" + (f" ({year})" if year else "")
