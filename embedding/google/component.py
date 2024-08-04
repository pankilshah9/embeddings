from typing import override

import google.generativeai as genai
import numpy as np
import pandas as pd  # Add this import statement
from typing_extensions import Dict

from embedding.embedding import EmbeddingsResponse, EmbeddingItem, EmbeddingsRequest, AbstractEmbeddingService
from embedding.google.config import api_config_schema
from utils.json_utils import validate_or_get_default_json


class EmbeddingService(AbstractEmbeddingService):
    """
    EmbeddingService is a class for creating text embeddings using Google Generative AI's embedding models.

    Supported Models:
    - models/embedding-001
    - models/text-embedding-004
    """

    def __init__(self, settings: Dict[str, any] = None):
        """
           Initializes the EmbeddingService with the provided configuration settings.

           :param settings: Optional dictionary containing configuration settings for the API.
        """

        super().__init__(settings)

        settings = validate_or_get_default_json(api_config_schema(), settings)

        genai.configure(api_key=settings.get("api_key"))
        self.model = settings.get("model")
        self.dimensions = settings.get("dimensions")
        self.encoding_format = settings.get("encoding_format")

    @override
    def create_embeddings_list(self, request: "EmbeddingsRequest") -> \
            "EmbeddingsResponse":

        inputs = []
        for data in request.data:
            inputs.append(data.content)

        embeddings = genai.embed_content(
            model=self.model,
            content=inputs,
            task_type="RETRIEVAL_DOCUMENT"
        )

        data = embeddings['embedding']
        embedding_items = []
        for index in range(len(data)):
            request_item = request.data[index]
            embedding_items.append(
                EmbeddingItem(content=request_item.content, embedding=data[index], input_type=request_item.input_type))

        return EmbeddingsResponse(embeddings=embedding_items)

    def generate_and_retrieve_search(self, query: str, base: pd.DataFrame) -> str:
        """
        Generates embeddings for a search query and retrieves the closest matching document from the base DataFrame.

        :param query: The search query string.
        :param base: DataFrame containing the documents with precomputed embeddings.
        :return: The content of the closest matching document.
        """
        query_embedding = genai.embed_content(
            model=self.model,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )['embedding']

        dot_product = np.dot(np.stack(base["Embeddings"]), query_embedding)
        index = np.argmax(dot_product)
        return base.iloc[index]["Content"]
