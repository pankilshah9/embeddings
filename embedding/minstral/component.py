from typing import override

from mistralai.client import MistralClient
from typing_extensions import Dict

from embedding.embedding import EmbeddingsResponse, EmbeddingItem, EmbeddingsRequest, AbstractEmbeddingService
from embedding.minstral.config import api_config_schema
from utils.json_utils import validate_or_get_default_json


class EmbeddingService(AbstractEmbeddingService):
    """
    EmbeddingService is a class for creating text embeddings using Mistral's embedding models.

    Supported Models:
    - mistral-embed
    """

    def __init__(self, settings: Dict[str, any] = None):
        """
          Initializes the EmbeddingService with the provided configuration settings.

          :param settings: Optional dictionary containing configuration settings for the API.
        """

        super().__init__(settings)
        settings = validate_or_get_default_json(api_config_schema(), settings)

        self.client = MistralClient(api_key=settings.get("api_key"))
        self.model = settings.get("model")

    @override
    def create_embeddings_list(self, request: EmbeddingsRequest) -> \
            EmbeddingsResponse:

        inputs = []
        for data in request.data:
            inputs.append(data.content)

        embeddings_batch_response = self.client.embeddings(
            model=self.model,
            input=inputs
        )

        data = embeddings_batch_response.data

        embedding_items = []

        for index in range(len(data)):
            request_item = request.data[index]
            embedding_items.append(
                EmbeddingItem(content=request_item.content, embedding=data[index].embedding,
                              input_type=request_item.input_type))

        return EmbeddingsResponse(embeddings=embedding_items)