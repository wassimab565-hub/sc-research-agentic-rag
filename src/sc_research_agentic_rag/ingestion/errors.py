# src\sc_research_agentic_rag\ingestion\
class IngestionError(Exception):
    """Raised when a scientific source cannot be queried or its response is invalid."""