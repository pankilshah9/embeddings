from typing import List

def generate_embeddings(data: List[str]):
    from component import EmbeddingService

    embedding_service = EmbeddingService()
    return embedding_service.create_embeddings_list(data).embeddings

if __name__ == "__main__":
    # Example test case
    example_texts = ["AI and machine learning", "AWS Bedrock embeddings"]
    embeddings = generate_embeddings(example_texts)
    for embedding in embeddings:
        print(f"Text: {embedding.text}, Embedding: {embedding.embedding[:10]}...")  # Print first 10 dimensions for brevity