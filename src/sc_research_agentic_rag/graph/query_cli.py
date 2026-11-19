# src\sc_research_agentic_rag\graph\
import argparse
import sys
from collections.abc import Sequence

from sc_research_agentic_rag.graph.errors import GraphError
from sc_research_agentic_rag.graph.queries import GraphQueries
from sc_research_agentic_rag.graph.store import GraphStore


def build_store() -> GraphStore:
    return GraphStore.from_settings()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the graph: who wrote about a concept, a paper's neighbours.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--concept", help="papers and authors mentioning a concept")
    group.add_argument("--author", help="papers and main concepts of an author (partial name allowed)")
    group.add_argument("--related", metavar="UID", help="papers sharing the most concepts with a paper")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    value = args.concept or args.author or args.related
    if not value or not value.strip():
        parser.error("the value must not be empty")

    store = None
    try:
        store = build_store()
        queries = GraphQueries(store)
        if args.concept:
            _concept(queries, args.concept, args.limit)
        elif args.author:
            _author(queries, args.author, args.limit)
        else:
            _related(queries, args.related, args.limit)
    except ValueError as exc:
        parser.error(str(exc))
    except GraphError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()
    return 0


def _label(title: str | None, uid: str, year: int | None) -> str:
    return f"{title or uid}" + (f" ({year})" if year else "")


def _names(names: Sequence[str], shown: int = 3) -> str:
    return ", ".join(names[:shown]) + (" et al." if len(names) > shown else "")


def _concept(queries: GraphQueries, name: str, limit: int) -> None:
    papers = queries.papers_for_concept(name, limit)
    if not papers:
        print(f"no paper mentions '{name}'")
        similar = queries.similar_concepts(name)
        if similar:
            print("similar concepts: " + ", ".join(f"{n} ({c} chunks)" for n, c in similar))
        return
    for rank, paper in enumerate(papers, start=1):
        pages = ", ".join(str(page) for page in paper.pages)
        print(f"{rank}. [{paper.uid}] {_label(paper.title, paper.uid, paper.year)}")
        print(f"   {paper.chunks} chunks, pages {pages}" + (f" | {_names(paper.authors)}" if paper.authors else ""))


def _author(queries: GraphQueries, name: str, limit: int) -> None:
    papers = queries.papers_for_author(name, limit)
    if not papers:
        print(f"no author matches '{name}'")
        return
    print("matched authors: " + ", ".join(sorted({a for paper in papers for a in paper.matched_authors})))
    for rank, paper in enumerate(papers, start=1):
        print(f"{rank}. [{paper.uid}] {_label(paper.title, paper.uid, paper.year)}")
    concepts = queries.author_concepts(name)
    if concepts:
        print("main concepts: " + ", ".join(f"{n} ({c})" for n, c in concepts))


def _related(queries: GraphQueries, uid: str, limit: int) -> None:
    papers = queries.related_papers(uid, limit)
    if not papers:
        print(f"no related paper for '{uid}' (unknown uid, or no extracted concepts)")
        return
    for rank, paper in enumerate(papers, start=1):
        print(f"{rank}. [{paper.uid}] {_label(paper.title, paper.uid, paper.year)} (score {paper.score:.2f})")
        print(f"   shared: {', '.join(paper.shared)}")