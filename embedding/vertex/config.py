import env_setup as env
from typing import Literal, Optional

# Configuration settings with default values
MODEL: str = "textembedding-gecko@001"  # Default model
PROJECT: Optional[str] = env.getenv('VERTEXAI_PROJECT')
LOCATION: str = env.getenv('VERTEXAI_LOCATION', 'us-central1')
REQUEST_PARALLELISM: int = int(env.getenv('VERTEXAI_REQUEST_PARALLELISM', 5))
MAX_RETRIES: int = int(env.getenv('VERTEXAI_MAX_RETRIES', 6))
API_KEY: Optional[str] = env.getenv('VERTEXAI_API_KEY')

# Supported models
SUPPORTED_MODELS = [
    "textembedding-gecko@001",
]