from typing import List, Optional, Literal, Tuple
import concurrent.futures
import logging
import string
import re

from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, Aborted, DeadlineExceeded, InvalidArgument
import google.auth
from vertexai.language_models import TextEmbeddingModel
import numpy as np

from config import MODEL, PROJECT, LOCATION, REQUEST_PARALLELISM, MAX_RETRIES, SUPPORTED_MODELS
from embedding.embedding import EmbeddingsResponse, EmbeddingItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    EmbeddingService is a class for creating text embeddings using Google's Vertex AI embedding models.

    Supported Models:
    - textembedding-gecko@001
    """

    def __init__(self, model_name: str = MODEL, project: Optional[str] = PROJECT, location: str = LOCATION,
                 request_parallelism: int = REQUEST_PARALLELISM, max_retries: int = MAX_RETRIES):
        """
        Initializes the EmbeddingService with the provided configuration settings.

        :param model_name: Model to be used for creating embeddings.
        :param project: Google Cloud project ID.
        :param location: Google Cloud location.
        :param request_parallelism: Number of concurrent requests to Vertex AI.
        :param max_retries: Max number of retries for failed requests.
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported. Supported models are: {', '.join(SUPPORTED_MODELS)}")

        self.model_name = model_name
        self.project = project
        self.location = location
        self.request_parallelism = request_parallelism
        self.max_retries = max_retries

        credentials, _ = google.auth.default()
        self.client = TextEmbeddingModel.from_pretrained(self.model_name, credentials=credentials)

    @staticmethod
    def _split_by_punctuation(text: str) -> List[str]:
        """Splits a string by punctuation and whitespace characters."""
        split_by = string.punctuation + '\t\n '
        pattern = f'([{split_by}])'
        return [segment for segment in re.split(pattern, text) if segment]

    @staticmethod
    def _prepare_batches(texts: List[str], batch_size: int) -> List[List[str]]:
        """Prepare text batches based on batch size and token limits."""
        text_index = 0
        texts_len = len(texts)
        batch_token_len = 0
        batches = []
        current_batch = []

        while text_index < texts_len:
            current_text = texts[text_index]
            current_text_token_count = len(EmbeddingService._split_by_punctuation(current_text)) * 2

            if current_text_token_count > 20000:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [current_text]
                text_index += 1
            elif batch_token_len + current_text_token_count > 20000 or len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
                batch_token_len = 0
            else:
                batch_token_len += current_text_token_count
                current_batch.append(current_text)
                text_index += 1

        if current_batch:
            batches.append(current_batch)

        return batches

    def _get_embeddings_with_retry(self, texts: List[str], task_type: Optional[str] = None) -> List[List[float]]:
        """Fetch embeddings with retry logic on failure."""
        errors = [ResourceExhausted, ServiceUnavailable, Aborted, DeadlineExceeded]
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                if task_type:
                    requests = [self.client.create_text_embedding_request(text, task_type) for text in texts]
                else:
                    requests = texts
                embeddings = self.client.get_embeddings(requests)
                return [embedding.values for embedding in embeddings]
            except tuple(errors) as e:
                logger.warning(f"Retrying due to {e}, attempt {retry_count + 1}/{self.max_retries}")
                retry_count += 1

        raise RuntimeError(f"Failed to fetch embeddings after {self.max_retries} retries")

    def create_embeddings_list(self, texts: List[str], batch_size: int = 250) -> EmbeddingsResponse:
        """Creates embeddings for a list of texts."""
        batches = self._prepare_batches(texts, batch_size)
        embeddings = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.request_parallelism) as executor:
            future_to_batch = {executor.submit(self._get_embeddings_with_retry, batch): batch for batch in batches}
            for future in concurrent.futures.as_completed(future_to_batch):
                embeddings.extend(future.result())

        embedding_items = [EmbeddingItem(text=texts[i], embedding=embedding) for i, embedding in enumerate(embeddings)]
        return EmbeddingsResponse(embeddings=embedding_items)

    def create_embedding(self, text: str) -> EmbeddingsResponse:
        """Creates an embedding for a single text."""
        return self.create_embeddings_list([text])


# Example usage
if __name__ == "__main__":
    embedding_service = EmbeddingService()
    sample_texts = ["Natural Language Processing with Vertex AI", "Embedding models using Google Cloud"]
    embeddings = embedding_service.create_embeddings_list(sample_texts)
    for item in embeddings.embeddings:
        print(
            f"Text: {item.text}, Embedding: {item.embedding[:10]}...")  # Print first 10 dimensions of the embedding for brevity