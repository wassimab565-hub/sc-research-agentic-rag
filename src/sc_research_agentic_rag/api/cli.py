# src\sc_research_agentic_rag\api\
import argparse
from collections.abc import Sequence

import uvicorn

from sc_research_agentic_rag.api.app import create_app
from sc_research_agentic_rag.api.config import ApiSettings


def main(argv: Sequence[str] | None = None) -> int:
    settings = ApiSettings()
    parser = argparse.ArgumentParser(description="Start the research API (one process, no reload: local Qdrant is single-process).")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0
