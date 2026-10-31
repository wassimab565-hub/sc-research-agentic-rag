import json

import httpx
import pytest
from qdrant_client import QdrantClient

from sc_research_agentic_rag.document_processing import PdfDownloader, process_papers, process_uploads
from sc_research_agentic_rag.document_processing.pipeline import chunks_path
from sc_research_agentic_rag.embeddings import EmbeddingError
from sc_research_agentic_rag.ingestion.models import Paper, PaperSource
from sc_research_agentic_rag.storage import IndexStatus, Indexer, VectorStore, VectorStoreError
from sc_research_agentic_rag.storage import cli

PAGES = ["alpha " * 120, "beta " * 120]


def vector_for(text: str) -> list[float]:
    if "alpha" in text:
        return [1.0, 0.0, 0.0]
    if "beta" in text:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


class FakeEmbedder:
    model_name = "fake/model"
    dimension = 3

    def __init__(self) -> None:
        self.calls = 0

    def embed_passages(self, texts):
        self.calls += 1
        return [vector_for(text) for text in texts]

    def embed_query(self, text):
        return vector_for(text)


def make_store() -> VectorStore:
    return VectorStore(QdrantClient(":memory:"), "chunks_fake")


def make_paper() -> Paper:
    return Paper(
        source=PaperSource.ARXIV,
        source_id="1",
        title="Paper One",
        authors=["Ann", "Bob"],
        url="https://arxiv.org/abs/1",
        pdf_url="https://x/p.pdf",
        doi="10.1/x",
    )


@pytest.fixture
def paper_file(tmp_path, write_pdf):
    pdf_bytes = write_pdf(PAGES, "source.pdf").read_bytes()
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=pdf_bytes)))
    downloader = PdfDownloader(output_dir=tmp_path / "pdfs", client=client)
    out = tmp_path / "chunks"
    process_papers([make_paper()], downloader, out, chunk_size=300, overlap=50)
    return chunks_path(out, "arxiv:1")


@pytest.fixture
def upload_file(tmp_path, write_pdf):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "My Report.pdf").write_bytes(write_pdf(PAGES, "tmp.pdf").read_bytes())
    out = tmp_path / "upload_chunks"
    process_uploads(uploads, out, chunk_size=300, overlap=50)
    return chunks_path(out, "upload:My Report")


def chunk_count(path) -> int:
    return len(json.loads(path.read_text(encoding="utf-8"))["chunks"])


def test_indexes_paper_with_metadata_and_provenance(paper_file) -> None:
    embedder, store = FakeEmbedder(), make_store()
    [result] = Indexer(embedder, store).index_files([paper_file])

    assert result.status == IndexStatus.INDEXED
    assert result.chunks == chunk_count(paper_file) > 2
    assert store.count() == result.chunks

    hits = store.search(embedder.embed_query("alpha"), limit=3)
    assert all("alpha" in hit.payload["text"] for hit in hits)
    payload = hits[0].payload
    assert payload["title"] == "Paper One"
    assert payload["authors"] == ["Ann", "Bob"]
    assert payload["url"] == "https://arxiv.org/abs/1"
    assert payload["pdf_url"] == "https://x/p.pdf"
    assert payload["doi"] == "10.1/x"
    assert payload["source"] == "arxiv"
    assert payload["paper_uid"] == "arxiv:1"
    assert payload["page"] == 1
    assert payload["embedding_model"] == "fake/model"
    assert hits[0].chunk_id.startswith("arxiv:1#")

    beta_hit = store.search(embedder.embed_query("beta"), limit=1)[0]
    assert beta_hit.payload["page"] == 2


def test_upload_payload_has_fallbacks(upload_file) -> None:
    store = make_store()
    Indexer(FakeEmbedder(), store).index_files([upload_file])
    payload = store.search([1.0, 0.0, 0.0], limit=1)[0].payload
    assert payload["title"] == "My Report"
    assert payload["source"] == "upload"
    assert payload["url"] is None
    assert payload["doi"] is None
    assert payload["authors"] == []


def test_already_indexed_paper_is_skipped_unless_forced(paper_file) -> None:
    embedder, store = FakeEmbedder(), make_store()
    indexer = Indexer(embedder, store)
    indexer.index_files([paper_file])
    assert embedder.calls == 1

    [again] = indexer.index_files([paper_file])
    assert again.status == IndexStatus.EXISTS
    assert embedder.calls == 1

    [forced] = indexer.index_files([paper_file], force=True)
    assert forced.status == IndexStatus.INDEXED
    assert embedder.calls == 2
    assert store.count() == forced.chunks


def test_reindexing_removes_stale_chunks(paper_file) -> None:
    store = make_store()
    indexer = Indexer(FakeEmbedder(), store)
    indexer.index_files([paper_file])
    assert store.count() > 1

    data = json.loads(paper_file.read_text(encoding="utf-8"))
    data["chunks"] = data["chunks"][:1]
    paper_file.write_text(json.dumps(data), encoding="utf-8")

    [result] = indexer.index_files([paper_file])
    assert result.status == IndexStatus.INDEXED
    assert store.count() == 1


def test_bad_files_fail_without_stopping_the_batch(tmp_path, paper_file) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    missing_key = tmp_path / "nokey.json"
    missing_key.write_text('{"paper_uid": "x"}', encoding="utf-8")

    results = Indexer(FakeEmbedder(), make_store()).index_files([bad, missing_key, paper_file])

    assert [r.status for r in results] == [IndexStatus.FAILED, IndexStatus.FAILED, IndexStatus.INDEXED]
    assert "invalid chunk file" in results[0].message


def test_empty_and_invalid_chunks_fail(tmp_path, paper_file) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"paper_uid": "x", "paper": None, "chunks": []}), encoding="utf-8")

    data = json.loads(paper_file.read_text(encoding="utf-8"))
    data["chunks"][0]["text"] = "   "
    blank = tmp_path / "blank.json"
    blank.write_text(json.dumps(data), encoding="utf-8")

    results = Indexer(FakeEmbedder(), make_store()).index_files([empty, blank])
    assert [r.status for r in results] == [IndexStatus.FAILED, IndexStatus.FAILED]
    assert "no chunks" in results[0].message


def test_dimension_mismatch_is_raised(paper_file) -> None:
    store = make_store()
    store.ensure_collection(5)
    with pytest.raises(VectorStoreError):
        Indexer(FakeEmbedder(), store).index_files([paper_file])


def test_embedding_errors_are_not_swallowed(paper_file) -> None:
    class BrokenEmbedder(FakeEmbedder):
        def embed_passages(self, texts):
            raise EmbeddingError("model exploded")

    with pytest.raises(EmbeddingError):
        Indexer(BrokenEmbedder(), make_store()).index_files([paper_file])


def test_cli_indexes_and_reports(monkeypatch, paper_file, upload_file, capsys, tmp_path) -> None:
    embedder, store = FakeEmbedder(), make_store()
    monkeypatch.setattr(cli, "build_services", lambda: (embedder, store))
    chunks_dir = tmp_path / "all"
    chunks_dir.mkdir()
    for source in (paper_file, upload_file):
        (chunks_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    code = cli.main(["--chunks-dir", str(chunks_dir)])

    printed = capsys.readouterr().out
    assert code == 0
    assert "[indexed] arxiv:1" in printed
    assert "[indexed] upload:My Report" in printed
    assert "indexed: 2" in printed
    assert "collection 'chunks_fake' now holds" in printed


def test_cli_second_run_reports_exists_and_force_reindexes(monkeypatch, paper_file, capsys) -> None:
    class SharedStore(VectorStore):
        def close(self) -> None:  # keep the in-memory data between CLI runs
            pass

    embedder = FakeEmbedder()
    store = SharedStore(QdrantClient(":memory:"), "chunks_fake")
    monkeypatch.setattr(cli, "build_services", lambda: (embedder, store))
    args = ["--chunks-dir", str(paper_file.parent)]

    cli.main(args)
    capsys.readouterr()
    cli.main(args)
    assert "[exists ] arxiv:1" in capsys.readouterr().out

    calls = embedder.calls
    cli.main([*args, "--force"])
    assert embedder.calls == calls + 1


def test_cli_without_chunk_files(tmp_path, capsys) -> None:
    assert cli.main(["--chunks-dir", str(tmp_path)]) == 1
    assert "no chunk files" in capsys.readouterr().err


def test_cli_returns_1_when_everything_fails(monkeypatch, tmp_path, capsys) -> None:
    (tmp_path / "bad.json").write_text("nope", encoding="utf-8")
    monkeypatch.setattr(cli, "build_services", lambda: (FakeEmbedder(), make_store()))
    assert cli.main(["--chunks-dir", str(tmp_path)]) == 1
    assert "[failed" in capsys.readouterr().out


def test_cli_reports_model_errors(monkeypatch, tmp_path, capsys) -> None:
    class BrokenEmbedder(FakeEmbedder):
        @property
        def dimension(self) -> int:
            raise EmbeddingError("cannot load model")

    (tmp_path / "x.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "build_services", lambda: (BrokenEmbedder(), make_store()))
    assert cli.main(["--chunks-dir", str(tmp_path)]) == 1
    assert "cannot load model" in capsys.readouterr().err