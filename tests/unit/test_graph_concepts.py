from sc_research_agentic_rag.graph.concepts import MAX_CONCEPTS, build_messages, parse_concepts

PASSAGE = "Dense retrieval uses a Dot Product between vectors. BM25 and DPR are baselines. Les modèles de langage."


def test_keeps_only_concepts_found_in_the_passage() -> None:
    raw = '["dense retrieval", "dot product", "unicorn model"]'
    assert parse_concepts(raw, PASSAGE) == ["dense retrieval", "dot product"]


def test_accepts_code_fences_and_surrounding_text() -> None:
    assert parse_concepts('Here:\n```json\n["BM25", "DPR"]\n```', PASSAGE) == ["BM25", "DPR"]


def test_matching_ignores_case_and_accents() -> None:
    assert parse_concepts('["modeles de langage"]', PASSAGE) == ["modeles de langage"]


def test_rejects_generic_short_duplicate_and_non_string_items() -> None:
    assert parse_concepts('["BM25", "bm25", "model", "ab", "12345", 3]', PASSAGE) == ["BM25"]


def test_caps_the_number_of_concepts() -> None:
    names = [f"term{n}" for n in range(20)]
    result = parse_concepts(str(names).replace("'", '"'), " ".join(names))
    assert len(result) == MAX_CONCEPTS


def test_empty_array_is_valid_but_garbage_is_not() -> None:
    assert parse_concepts("[]", PASSAGE) == []
    for raw in ("no json", '{"a": 1}', "[unclosed"):
        assert parse_concepts(raw, PASSAGE) is None


def test_messages_contain_the_passage() -> None:
    system, user = build_messages(PASSAGE)
    assert system.role == "system"
    assert PASSAGE in user.content

def test_rejects_phrases_with_articles_and_math_symbols() -> None:
    passage = "retrieve a list of documents with ˆst and y<t and BM25"
    raw = '["retrieve a list of documents", "ˆst", "y<t", "BM25"]'
    assert parse_concepts(raw, passage) == ["BM25"]