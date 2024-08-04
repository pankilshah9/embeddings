from typing import List

def generate_embeddings(data: List[str]):
    from component import EmbeddingService

    embedding_service = EmbeddingService()
    return embedding_service.create_embeddings_list(data).embeddings

# Example test case
if __name__ == "__main__":
    example_texts = ["AI and machine learning", "Vertex AI embeddings"]
    embeddings = generate_embeddings(example_texts)
    print(embeddings)