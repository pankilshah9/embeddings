from typing import List, Dict
from typing import override

import cohere
import numpy as np

from embedding.cohere.config import api_config_schema
from embedding.embedding import EmbeddingsResponse, EmbeddingItem, EmbeddingsRequest, AbstractEmbeddingService
from utils.json_utils import validate_or_get_default_json


class EmbeddingService(AbstractEmbeddingService):
    """
    EmbeddingService is a class for creating text embeddings using Cohere's embedding models.

    Supported Models:
    - embed-english-v3.0
    - embed-multilingual-v3.0
    - embed-english-light-v3.0
    - embed-multilingual-light-v3.0
    - embed-english-v2.0
    - embed-english-light-v2.0
    - embed-multilingual-v2.0
    """

    # def __init__(self, api_key: str = API_KEY, model: str = MODEL, input_type: str = INPUT_TYPE,
    #             embedding_type: str = EMBEDDING_TYPE, truncate: str = "END"):
    def __init__(self, settings: Dict[str, any] = None):
        """
           Initializes the EmbeddingService with the provided configuration settings.

           :param settings: Optional dictionary containing configuration settings for the API.
        """

        super().__init__(settings)
        settings = validate_or_get_default_json(api_config_schema(), settings)

        self.client = cohere.Client(api_key=settings.get("api_key"))
        self.model = settings.get("model")
        self.input_type = settings.get("input_type")
        self.embedding_type = settings.get("encoding_format")
        self.truncate = "END"

    @override
    def create_embeddings_list(self, request: "EmbeddingsRequest") -> \
            "EmbeddingsResponse":

        input_data = []
        for item in request.data:
            if item.input_type == "text" or item.input_type == "image":
                input_data.append(item.content)
            else:
                raise ValueError(f"Type {item.input_type} not supported")

        response = self.client.embed(
            texts=input_data,
            model=self.model,
            input_type=self.input_type,
            embedding_types=[self.embedding_type],
            truncate=self.truncate
        )

        embeddings = getattr(response.embeddings, self.get_embedding_type(self.embedding_type))

        embedding_items = []

        for index in range(len(embeddings)):
            request_item = request.data[index]
            embedding_items.append(
                EmbeddingItem(content=request_item.content, embedding=embeddings[index],
                              input_type=request_item.input_type))

            return EmbeddingsResponse(embeddings=embedding_items)

    @staticmethod
    def calculate_similarity(a: List[float], b: List[float]) -> float:
        """
        Calculates the cosine similarity between two embeddings.

        :param a: First embedding vector.
        :param b: Second embedding vector.
        :return: Cosine similarity score.
        """
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_embedding_type(self, embedding_type: str):
        if embedding_type == "float_":
            return "float"

        return embedding_type
