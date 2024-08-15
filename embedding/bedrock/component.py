import os
import json
import boto3
import numpy as np
from loguru import logger
from typing import List, Dict, Any
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from config import MODEL, REGION_NAME, CREDENTIALS_PROFILE_NAME, ENDPOINT_URL, NORMALIZE, SUPPORTED_MODELS, \
    api_config_schema
from embedding.embedding import EmbeddingsResponse, EmbeddingItem, AbstractEmbeddingService, EmbeddingsRequest
from utils.json_utils import validate_or_get_default_json


class EmbeddingService(AbstractEmbeddingService):

    def __init__(self, settings: Dict[str, Any] = None):
        super().__init__(settings)

        settings = validate_or_get_default_json(api_config_schema(), settings)

        self.model = settings.get("model")
        self.region_name = settings.get("region_name", REGION_NAME)
        self.credentials_profile_name = settings.get("credentials_profile_name", CREDENTIALS_PROFILE_NAME)
        self.endpoint_url = settings.get("endpoint_url", ENDPOINT_URL)
        self.normalize = settings.get("normalize", NORMALIZE)
        self.client = self._create_client()

    def _create_client(self):
        """Create a client to connect to Bedrock"""
        try:
            session = boto3.Session(profile_name=self.credentials_profile_name)
            client_params = {"region_name": self.region_name}
            if self.endpoint_url:
                client_params["endpoint_url"] = self.endpoint_url

            return session.client('bedrock-runtime', **client_params)
        except (NoCredentialsError, PartialCredentialsError):
            logger.error(f"Could not load credentials to authenticate with AWS client.")
            raise
        except Exception:
            logger.error(f"Could not connect to Bedrock")
            raise

    @staticmethod
    def _normalize_vector(embeddings: List[float]) -> List[float]:
        """Normalize the embedding to a unit vector."""
        emb = np.array(embeddings)
        norm_emb = emb / np.linalg.norm(emb)
        return norm_emb.tolist()

    def _embedding_func(self, text: str) -> List[float]:
        """Call out to Bedrock embedding endpoint"""
        try:
            text = text.replace(os.linesep, " ")

            provider = self.model.split(".")[0]
            if provider == "cohere":
                input_body = {"texts": [text], "input_type": "search_document"}
            else:
                input_body = {"inputText": text}
            body = json.dumps(input_body)
            response = self.client.invoke_model(
                body=body,
                modelId=self.model,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())
            embedding = response_body.get('embeddings')[0] if provider == "cohere" else response_body.get("embedding")
            return embedding
        except Exception:
            logger.error(f"Error raised by inference endpoint.")
            raise

    def create_embeddings_list(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Create embeddings for the input text."""
        inputs = []
        for data in request.data:
            inputs.append(data.content)

        embedding_items = []
        for text in inputs:
            embedding = self._embedding_func(text)
            if self.normalize:
                embedding = self._normalize_vector(embedding)
            embedding_items.append(EmbeddingItem(content=text, embedding=embedding))

        return EmbeddingsResponse(embeddings=embedding_items)