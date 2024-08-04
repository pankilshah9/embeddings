from typing import List, Optional, Dict, Any
import os

from transformers import pipeline
from config import API_KEY, MODEL, CACHE_FOLDER, SUPPORTED_MODELS
from embedding.embedding import EmbeddingsResponse, EmbeddingItem


class EmbeddingService:
    """
    EmbeddingService is a class for creating text embeddings using HuggingFace's transformer models.

    Supported Models:
    - sentence-transformers/all-mpnet-base-v2
    - hkunlp/instructor-large
    - BAAI/bge-large-en
    """

    def __init__(self, model: str = MODEL, cache_folder: Optional[str] = CACHE_FOLDER, api_key: str = API_KEY):
        """
        Initializes the EmbeddingService with the provided configuration settings.

        :param model: Model to be used for creating embeddings.
        :param cache_folder: Folder path to cache the models.
        :param api_key: API key for the HuggingFace service.
        """
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model}' is not supported. Supported models are: {', '.join(SUPPORTED_MODELS)}")

        os.environ['HF_HOME'] = cache_folder or os.path.expanduser("~/.cache/huggingface")
        self.model = model
        self.client = pipeline("feature-extraction", model=model, cache_dir=cache_folder, use_auth_token=api_key)

    def create_embeddings_list(self, texts: List[str]) -> EmbeddingsResponse:
        """
        Creates embeddings for a list of texts.

        :param texts: A list of strings for which embeddings are to be created.
        :return: A list of embeddings.
        """
        embeddings = self.client(texts)
        embedding_items = []
        for index, embedding in enumerate(embeddings):
            flattened_embedding = [val for sublist in embedding for val in
                                   sublist]  # Flatten the multi-dimensional embeddings
            embedding_items.append(EmbeddingItem(text=texts[index], embedding=flattened_embedding))

        return EmbeddingsResponse(embeddings=embedding_items)

    def create_embedding(self, text: str) -> EmbeddingsResponse:
        """
        Creates an embedding for a single text.

        :param text: A string for which the embedding is to be created.
        :return: A list of floats representing the embedding.
        """
        return self.create_embeddings_list([text])


# Example usage
if __name__ == "__main__":
    embedding_service = EmbeddingService()
    phrases = ["i love soup", "soup is my favorite", "london is far away"]
    embeddings = embedding_service.create_embeddings_list(phrases)
    print(embeddings)
    soup1, soup2, london = embeddings.embeddings[0], embeddings.embeddings[1], embeddings.embeddings[2]

    print(f"Embeddings for 'i love soup': {soup1}")
    print(f"Embeddings for 'soup is my favorite': {soup2}")
    print(f"Embeddings for 'london is far away': {london}")