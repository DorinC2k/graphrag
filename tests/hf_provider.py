import asyncio
from sentence_transformers import SentenceTransformer
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.factory import ModelFactory


class HuggingFaceEmbeddingModel:
    """Embedding model wrapper around a Hugging Face SentenceTransformer."""

    def __init__(self, *, name: str, config: LanguageModelConfig, **kwargs):
        self.model = SentenceTransformer(config.model)
        self.config = config

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        embeddings = self.model.encode(text_list, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def embed(self, text: str, **kwargs) -> list[float]:
        return self.embed_batch([text])[0]

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_batch, text_list)

    async def aembed(self, text: str, **kwargs) -> list[float]:
        result = await self.aembed_batch([text])
        return result[0]


def register_huggingface_embedding() -> None:
    """Register the Hugging Face embedding provider with the ModelFactory."""
    ModelFactory.register_embedding(
        "huggingface_embedding",
        lambda **kwargs: HuggingFaceEmbeddingModel(**kwargs),
    )
