import env_setup as env
from typing import Literal

# Configuration settings with default values
MODEL: str = "sentence-transformers/all-mpnet-base-v2"  # Default model
CACHE_FOLDER: str = None  # Default cache folder
EMBEDDING_TYPE: Literal["float"] = "float"
API_KEY: str = env.getenv('HUGGINGFACE_API_KEY')

# Supported models
SUPPORTED_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "hkunlp/instructor-large",
    "BAAI/bge-large-en"
]