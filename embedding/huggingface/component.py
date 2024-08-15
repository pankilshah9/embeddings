import requests
from loguru import logger
from typing import Dict, Any
from config import API_KEY, MODEL, api_config_schema, API_URL
from embedding.embedding import EmbeddingsResponse, EmbeddingItem, AbstractEmbeddingService, EmbeddingsRequest
from utils.json_utils import validate_or_get_default_json


class EmbeddingService(AbstractEmbeddingService):

    def __init__(self, settings: Dict[str, Any] = None):
        super().__init__(settings)

        settings = validate_or_get_default_json(api_config_schema(), settings)
        self.model = settings.get('model', MODEL)
        self.api_key = settings.get('api_key', API_KEY)
        self.api_url = settings.get('api_url', API_URL)
        self.default_api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
        self._headers = {"Authorization": f"Bearer {self.api_key}"}

    def create_embeddings_list(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Create embeddings for the input text."""
        inputs = []
        for data in request.data:
            inputs.append(data.content)

        try:
            response = requests.post(
                self.api_url or self.default_api_url,
                headers=self._headers,
                json={
                    "inputs": inputs,
                    "options": {"wait_for_model": True, "use_cache": True},
                },
            )
            embeddings = response.json()
        except Exception:
            logger.error(f"Failed to create embeddings for {inputs}")
            raise

        embedding_items = []
        for index, embedding in enumerate(embeddings):
            request_item = request.data[index]
            embedding_items.append(
                EmbeddingItem(content=request_item.content, embedding=embedding, input_type=request_item.input_type)
            )

        return EmbeddingsResponse(embeddings=embedding_items)
