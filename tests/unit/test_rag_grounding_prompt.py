from sc_research_agentic_rag.llm import ChatMessage
from sc_research_agentic_rag.rag import SYSTEM_PROMPT, SourcePassage, build_messages
from sc_research_agentic_rag.rag.prompts import REMINDER


def passage() -> SourcePassage:
    return SourcePassage(
        number=1, paper_uid="a", title="T", authors=(), year=None, doi=None, url=None, source="arxiv",
        page=1, text="Some text.", chunk_ids=("a#0",), score=0.9,
    )


def test_system_prompt_forbids_answering_from_memory_and_unsupported_citations() -> None:
    assert "do not answer from memory" in SYSTEM_PROMPT
    assert "only when that source explicitly states it" in SYSTEM_PROMPT
    assert "do not cite any source" in SYSTEM_PROMPT


def test_reminder_comes_last_in_the_user_message() -> None:
    user = build_messages("What is X?", [passage()])[1].content
    assert user.endswith(REMINDER)
    assert user.index("Question: What is X?") < user.index(REMINDER)


def test_messages_are_chat_messages() -> None:
    assert all(isinstance(m, ChatMessage) for m in build_messages("q?", [passage()]))