from sc_research_agentic_rag.ingestion.arxiv_client import ArxivClient
from sc_research_agentic_rag.ingestion.dedup import deduplicate
from sc_research_agentic_rag.ingestion.errors import IngestionError
from sc_research_agentic_rag.ingestion.models import Paper, PaperSource
from sc_research_agentic_rag.ingestion.openalex_client import OpenAlexClient

__all__ = [
    "ArxivClient",
    "IngestionError",
    "OpenAlexClient",
    "Paper",
    "PaperSource",
    "deduplicate",
]