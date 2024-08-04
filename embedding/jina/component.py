from typing import Dict, override

import requests

from embedding.embedding import EmbeddingsResponse, EmbeddingItem, AbstractEmbeddingService, EmbeddingsRequest
from embedding.jina.config import api_config_schema
from utils.json_utils import validate_or_get_default_json


class EmbeddingService(AbstractEmbeddingService):
    """
    EmbeddingService is a class for creating text and image embeddings using Jina AI's embedding models.

    Supported Models:
    - jina-clip-v1
    """

    def __init__(self, settings: Dict[str, any] = None):

        """
         Initializes the EmbeddingService with the provided configuration settings.

         :param settings: Optional dictionary containing configuration settings for the API.
        """

        super().__init__(settings)

        settings = validate_or_get_default_json(api_config_schema(), settings)

        self.api_key = settings.get("api_key")
        self.model = settings.get("model")
        self.dimensions = settings.get("dimensions")
        self.encoding_format = settings.get("encoding_format")
        self.endpoint = "https://api.jina.ai/v1/embeddings"

    @override
    def create_embeddings_list(self, request: EmbeddingsRequest) -> EmbeddingsResponse:

        input_data = []
        for item in request.data:
            if item.input_type == "text" or item.input_type == "image":
                input_data.append({item.input_type: item.content})
            else:
                raise ValueError(f"Type {item.input_type} not supported")

        payload = {
            "model": self.model,
            "embedding_type": self.encoding_format,
            "input": input_data
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()

        json = response.json()
        data = json["data"]

        embedding_items = []

        for index in range(len(data)):
            request_item = request.data[index]
            embedding_items.append(
                EmbeddingItem(content=request_item.content, embedding=data[index]["embedding"],
                              input_type=request_item.input_type))

        return EmbeddingsResponse(embeddings=embedding_items)