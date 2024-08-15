from typing import List

from embedding.embedding import EmbeddingRequestItem, EmbeddingsRequest


def generate_embeddings(content: List[str]):
    from embedding.embedding import AbstractEmbeddingService

    data: List["EmbeddingRequestItem"] = []

    for item in content:
        data.append(EmbeddingRequestItem(content=item))

    return AbstractEmbeddingService.generate_embeddings("huggingface", EmbeddingsRequest(data=data)).embeddings


# Example test case
print(generate_embeddings(["AI", "Artificial Intelligence", "Machine Learning"]))