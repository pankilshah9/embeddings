import env_setup as env
from typing import Optional

# Configuration settings
MODEL: str = "amazon.titan-embed-text-v1"  # Default model
REGION_NAME: Optional[str] = env.getenv('AWS_REGION_NAME', 'us-east-1')
CREDENTIALS_PROFILE_NAME: Optional[str] = env.getenv('AWS_CREDENTIALS_PROFILE_NAME', 'default')
ENDPOINT_URL: Optional[str] = env.getenv('AWS_ENDPOINT_URL')
NORMALIZE: bool = env.getenv('NORMALIZE', 'False').lower() in ('true', '1', 't', 'y', 'yes')

# Supported models
SUPPORTED_MODELS = [
    "amazon.titan-embed-text-v1",
]