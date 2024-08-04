from typing import List
from embedding.embedding import EmbeddingsRequest, EmbeddingRequestItem


def generate_embeddings(content: List[str]):
    from embedding.embedding import AbstractEmbeddingService

    data: List["EmbeddingRequestItem"] = []

    for item in content:
        data.append(EmbeddingRequestItem(content=item))

    return AbstractEmbeddingService.generate_embeddings("google", EmbeddingsRequest(data=data)).embeddings


print(generate_embeddings(["Price of Maruti Suzuki India (MARUTI): ₹8901.23"]))
