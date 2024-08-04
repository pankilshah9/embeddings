from typing import override

from openai import OpenAI
from typing_extensions import Dict

from embedding.embedding import EmbeddingsResponse, EmbeddingItem, EmbeddingsRequest, AbstractEmbeddingService
from embedding.openai.config import api_config_schema
from utils.json_utils import validate_or_get_default_json


class EmbeddingService(AbstractEmbeddingService):
    """
    EmbeddingService is a class for creating text embeddings using OpenAI's embedding models.

    Supported Models:
    - text-embedding-3-large
    - text-embedding-3-small
    - text-embedding-ada-002
    """

    def __init__(self, settings: Dict[str, any] = None):
        """
       Initializes the EmbeddingService with the provided configuration settings.

       :param settings: Optional dictionary containing configuration settings for the API.
        """

        super().__init__(settings)

        settings = validate_or_get_default_json(api_config_schema(), settings)

        model = settings.get("model")

        self.client = OpenAI(api_key=settings.get("api_key"))
        self.model = model
        self.dimensions = settings.get("dimensions")
        self.encoding_format = settings.get("encoding_format")

    @override
    def create_embeddings_list(self, request: EmbeddingsRequest) -> \
            EmbeddingsResponse:
        inputs = []
        for data in request.data:
            inputs.append(data.content)

        embeddings = self.client.embeddings.create(
            model=self.model,
            input=inputs,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions
        )

        embedding_items = []
        data = embeddings.data

        for index in range(len(data)):
            request_item = request.data[index]
            embedding_items.append(
                EmbeddingItem(content=request_item.content, embedding=data[index].embedding,
                              input_type=request_item.input_type))

        return EmbeddingsResponse(embeddings=embedding_items)
