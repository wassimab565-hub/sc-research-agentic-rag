from collections.abc import Callable, Sequence
from typing import Any

from sc_research_agentic_rag.embeddings.config import EmbeddingSettings
from sc_research_agentic_rag.embeddings.errors import EmbeddingError


class Embedder:
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        batch_size: int = 16,
        query_prefix: str = "",
        passage_prefix: str = "",
        device: str | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._model_name = model_name
        self._revision = revision
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._device = device
        self._model_loader = model_loader or self._load_sentence_transformer
        self._model: Any = None
        self._dimension: int | None = None

    @classmethod
    def from_settings(cls, settings: EmbeddingSettings | None = None) -> "Embedder":
        settings = settings or EmbeddingSettings()
        return cls(
            model_name=settings.model,
            revision=settings.revision,
            batch_size=settings.batch_size,
            query_prefix=settings.query_prefix,
            passage_prefix=settings.passage_prefix,
            device=settings.device,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._encode(["dimension probe"])
        assert self._dimension is not None
        return self._dimension

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([self._passage_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("query must not be empty")
        return self._encode([self._query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result = vectors.tolist()
        self._dimension = len(result[0])
        return result

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                self._model = self._model_loader()
            except (OSError, ValueError, ImportError) as exc:
                raise EmbeddingError(f"cannot load embedding model '{self._model_name}': {exc}") from exc
        return self._model

    def _load_sentence_transformer(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self._model_name, revision=self._revision, device=self._device)