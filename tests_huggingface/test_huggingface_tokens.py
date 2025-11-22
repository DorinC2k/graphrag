import pytest

pytest.importorskip("tiktoken")

from graphrag.language_model.providers.huggingface.models import (
    HuggingFaceEmbeddingModel,
)


class DummyConfig:
    def __init__(
        self,
        *,
        api_base: str,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoding_model: str = "cl100k_base",
        api_key: str | None = None,
    ) -> None:
        self.type = "huggingface_embedding"
        self.model = model
        self.encoding_model = encoding_model
        self.api_base = api_base
        self.api_key = api_key


def test_huggingface_model_uses_primary_env_token(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "hf_primary")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_hub")
    monkeypatch.setenv("HUGGING_FACE_TOKEN_READ_KEY", "hf_legacy")

    config = DummyConfig(api_base="https://example.endpoint")

    model = HuggingFaceEmbeddingModel(name="hf", config=config)

    assert model.api_key == "hf_primary"


def test_huggingface_model_strips_bearer_prefix(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "Bearer hf_secret")

    config = DummyConfig(api_base="https://example.endpoint")

    model = HuggingFaceEmbeddingModel(name="hf", config=config)

    assert model.api_key == "hf_secret"


def test_huggingface_model_falls_back_to_hub_env_token(monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_API_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_hub")

    config = DummyConfig(api_base="https://example.endpoint")

    model = HuggingFaceEmbeddingModel(name="hf", config=config)

    assert model.api_key == "hf_hub"


def test_huggingface_model_respects_token_limit(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_TPR", "5")
    captured_inputs: list[list[str]] = []

    class DummyEncoding:
        def encode(self, text: str, allowed_special=None, disallowed_special=None):
            return text.split()

        def decode(self, tokens):
            return " ".join(tokens)

    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: DummyEncoding())
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda model: DummyEncoding())

    def fake_post(url, headers, json, timeout):
        captured_inputs.append(json["inputs"])

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embeddings": [[float(idx)] for idx, _ in enumerate(json["inputs"])]}

        return FakeResponse()

    monkeypatch.setattr("graphrag.language_model.providers.huggingface.models.requests", type("Requests", (), {"post": staticmethod(fake_post)}))

    config = DummyConfig(api_base="https://example.endpoint")

    model = HuggingFaceEmbeddingModel(name="hf", config=config)

    long_text = "sample text " * 50
    embeddings = model.embed_batch([long_text])

    assert len(captured_inputs) == 1
    assert len(captured_inputs[0]) > 1
    assert embeddings == [[1.0]]


def test_huggingface_model_defaults_remote_token_limit(monkeypatch):
    monkeypatch.delenv("GRAPHRAG_EMBEDDING_TPR", raising=False)
    captured_inputs: list[list[str]] = []

    class DummyEncoding:
        def encode(self, text: str, allowed_special=None, disallowed_special=None):
            return text.split()

        def decode(self, tokens):
            return " ".join(tokens)

    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: DummyEncoding())
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda model: DummyEncoding())

    def fake_post(url, headers, json, timeout):
        captured_inputs.append(json["inputs"])

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embeddings": [[float(idx)] for idx, _ in enumerate(json["inputs"])]}

        return FakeResponse()

    monkeypatch.setattr(
        "graphrag.language_model.providers.huggingface.models.requests",
        type("Requests", (), {"post": staticmethod(fake_post)}),
    )

    config = DummyConfig(api_base="https://example.endpoint")

    model = HuggingFaceEmbeddingModel(name="hf", config=config)

    long_text = "sample" + " sample" * 300
    embeddings = model.embed_batch([long_text])

    assert len(captured_inputs) == 1
    assert len(captured_inputs[0]) > 1
    assert all(len(item.split()) <= 256 for item in captured_inputs[0])
    assert embeddings == [[1.0]]
