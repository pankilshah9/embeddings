from typing import List

def generate_embeddings(data: List[str]):
    from component import EmbeddingService

    embedding_service = EmbeddingService()
    return embedding_service.create_embeddings_list(data).embeddings

# Example test case
print(generate_embeddings(["AI", "Artificial Intelligence", "Machine Learning"]))