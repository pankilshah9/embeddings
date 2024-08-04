import os
import json
from typing import List, Optional, Dict, Any
import numpy as np
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import boto3

from config import MODEL, REGION_NAME, CREDENTIALS_PROFILE_NAME, ENDPOINT_URL, NORMALIZE, SUPPORTED_MODELS
from embedding.embedding import EmbeddingsResponse, EmbeddingItem


class EmbeddingService:
    """
    EmbeddingService is a class for creating text embeddings using AWS Bedrock embedding models.

    Supported Models:
    - amazon.titan-embed-text-v1
    """

    def __init__(self, model_id: str = MODEL, region_name: Optional[str] = REGION_NAME,
                 credentials_profile_name: Optional[str] = CREDENTIALS_PROFILE_NAME,
                 endpoint_url: Optional[str] = ENDPOINT_URL, normalize: bool = NORMALIZE):
        """
        Initializes the EmbeddingService with the provided configuration settings.

        :param model_id: Model to be used for creating embeddings.
        :param region_name: AWS region name.
        :param credentials_profile_name: Profile name in the AWS credentials/config file.
        :param endpoint_url: Endpoint URL for the Bedrock service.
        :param normalize: Whether to normalize the embeddings to unit vectors.
        """
        if model_id not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_id}' is not supported. Supported models are: {', '.join(SUPPORTED_MODELS)}")

        self.model_id = model_id
        self.region_name = region_name
        self.credentials_profile_name = credentials_profile_name
        self.endpoint_url = endpoint_url
        self.normalize = normalize

        session = boto3.Session(profile_name=self.credentials_profile_name)
        client_params = {"region_name": self.region_name}
        if self.endpoint_url:
            client_params["endpoint_url"] = self.endpoint_url

        self.client = session.client('bedrock-runtime', **client_params)

    def _embedding_func(self, text: str) -> List[float]:
        """
        Call out to Bedrock embedding endpoint to get embeddings for a single text.

        :param text: The text to embed.
        :return: List of floats representing the embedding.
        """
        text = text.replace(os.linesep, " ")

        provider = self.model_id.split(".")[0]
        input_body = {}
        input_body["inputText"] = text if provider == "amazon" else {"texts": [text], "input_type": "search_document"}

        response = self.client.invoke_model(
            body=json.dumps(input_body),
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response.get("body").read())
        embedding = response_body.get("embedding") if provider == "amazon" else response_body.get("embeddings")[0]

        return embedding

    def _normalize_vector(self, embeddings: List[float]) -> List[float]:
        """
        Normalize the embedding to a unit vector.

        :param embeddings: List of floats representing the embedding.
        :return: Normalized embedding.
        """
        emb = np.array(embeddings)
        norm_emb = emb / np.linalg.norm(emb)
        return norm_emb.tolist()

    def create_embeddings_list(self, texts: List[str]) -> EmbeddingsResponse:
        """
        Creates embeddings for a list of texts.

        :param texts: List of strings to generate embeddings.
        :return: EmbeddingsResponse containing list of embeddings.
        """
        embedding_items = []
        for text in texts:
            embedding = self._embedding_func(text)
            if self.normalize:
                embedding = self._normalize_vector(embedding)

            embedding_items.append(EmbeddingItem(text=text, embedding=embedding))

        return EmbeddingsResponse(embeddings=embedding_items)

    def create_embedding(self, text: str) -> EmbeddingsResponse:
        """
        Creates an embedding for a single text.

        :param text: The text to embed.
        :return: EmbeddingsResponse containing the embedding.
        """
        return self.create_embeddings_list([text])


# Example usage
if __name__ == "__main__":
    embedding_service = EmbeddingService()
    sample_texts = ["Natural Language Processing with AWS Bedrock", "Embedding models using Amazon services"]
    embeddings = embedding_service.create_embeddings_list(sample_texts)
    for item in embeddings.embeddings:
        print(f"Text: {item.text}, Embedding: {item.embedding[:10]}...")  # Print first 10 dimensions for brevity