import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from sc_research_agentic_rag.ingestion.models import Paper


class PaperGraph(Protocol):
    def ensure_schema(self) -> None: ...
    def clear(self) -> None: ...
    def upsert_paper(self, paper: Paper) -> None: ...


@dataclass(frozen=True)
class BuildResult:
    papers: int
    uploads_skipped: int
    failed: dict[str, str] = field(default_factory=dict)


def read_chunk_papers(chunks_dir: Path) -> tuple[list[Paper], int, dict[str, str]]:
    papers: list[Paper] = []
    uploads = 0
    failed: dict[str, str] = {}
    for path in sorted(chunks_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("paper") is None:
                uploads += 1
                continue
            papers.append(Paper.model_validate(data["paper"]))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            failed[path.name] = str(exc)
    return papers, uploads, failed


def build_graph(chunks_dir: Path, graph: PaperGraph, force: bool = False) -> BuildResult:
    papers, uploads, failed = read_chunk_papers(chunks_dir)
    graph.ensure_schema()
    if force:
        graph.clear()
    for paper in papers:
        graph.upsert_paper(paper)
    return BuildResult(papers=len(papers), uploads_skipped=uploads, failed=failed)