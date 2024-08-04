from typing import List

import pydantic
from typing_extensions import Literal, Dict
import importlib

embeddings_api = Literal["bedrock", "cohere", "google", "huggingface", "jina", "minstral", "openai", "vertex", "voyage"]


class AbstractEmbeddingService:
    def __init__(self, settings: Dict[str, any] = None):
        pass

    def create_embeddings_list(self, request: "EmbeddingsRequest") -> \
            "EmbeddingsResponse":
        """
          Abstract method to create a list of embeddings based on the request.

          :param request: EmbeddingsRequest object containing data for embedding.
          :return: EmbeddingsResponse object containing the generated embeddings.
          :raises NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError("Subclasses must implement 'create_embeddings_list' method")

    @staticmethod
    def generate_embeddings(embedding_api: embeddings_api, request: "EmbeddingsRequest",
                            api_settings: Dict[str, any] = None) -> \
            "EmbeddingsResponse":
        """
          Generate embeddings using the specified embedding API.

          :param embedding_api: The embedding API to use (e.g., 'openai', 'google').
          :param request: EmbeddingsRequest object containing data for embedding.
          :param api_settings: Optional dictionary containing settings for the embedding API.
          :return: EmbeddingsResponse object containing the generated embeddings.
          :raises ValueError: If the embedding API is not found or is invalid.
        """

        module = importlib.import_module(f"embedding.{embedding_api}.component")
        EmbeddingService = getattr(module, "EmbeddingService")

        if issubclass(EmbeddingService, AbstractEmbeddingService):
            # Implement settings
            em: AbstractEmbeddingService = EmbeddingService(api_settings)
            return em.create_embeddings_list(request)

        raise ValueError(f"Embedding {embeddings_api} not found")


class EmbeddingsResponse(pydantic.BaseModel):
    embeddings: List["EmbeddingItem"]


supported_input_type = Literal["image", "text"]


class EmbeddingItem(pydantic.BaseModel):
    content: str
    embedding: List[float]
    input_type: supported_input_type = "text"


class EmbeddingsRequest(pydantic.BaseModel):
    data: List["EmbeddingRequestItem"]


class EmbeddingRequestItem(pydantic.BaseModel):
    content: str
    input_type: supported_input_type = "text"
