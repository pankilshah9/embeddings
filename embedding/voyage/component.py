from typing import override

import voyageai
from typing_extensions import Dict

from embedding.embedding import EmbeddingsResponse, EmbeddingItem, EmbeddingsRequest, AbstractEmbeddingService
from embedding.voyage.config import api_config_schema
from utils.json_utils import validate_or_get_default_json


class EmbeddingService(AbstractEmbeddingService):
    """
    EmbeddingService is a class for creating text embeddings using Voyage AI's embedding models.

    Supported Models:
    - voyage-large-2-instruct
    - voyage-finance-2
    - voyage-multilingual-2
    - voyage-law-2
    - voyage-code-2
    - voyage-large-2
    - voyage-2
    """

    def __init__(self, settings: Dict[str, any] = None):
        super().__init__(settings)
        settings = validate_or_get_default_json(api_config_schema(), settings)

        self.client = voyageai.Client(api_key=settings.get("api_key"))
        self.model = settings.get("model")
        self.input_type = settings.get("input_type")
        self.truncation = settings.get("truncation")

    @override
    def create_embeddings_list(self, request: EmbeddingsRequest) -> \
            EmbeddingsResponse:

        inputs = []
        for data in request.data:
            inputs.append(data.content)

        embeddings = self.client.embed(
            texts=inputs,
            model=self.model,
            input_type=self.input_type,
            truncation=self.truncation
        )

        embedding_items = []
        data = embeddings.embeddings

        for index in range(len(data)):
            request_item = request.data[index]
            embedding_items.append(
                EmbeddingItem(content=request_item.content, embedding=data[index],
                              input_type=request_item.input_type))

        return EmbeddingsResponse(embeddings=embedding_items)
