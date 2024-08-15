import env_setup as env
from typing import Optional

# Configuration settings
MODEL: str = "amazon.titan-embed-text-v1"  # Default model
REGION_NAME: Optional[str] = env.getenv('AWS_REGION_NAME', 'us-east-1')
CREDENTIALS_PROFILE_NAME: Optional[str] = env.getenv('AWS_PROFILE', 'default')
ENDPOINT_URL: Optional[str] = env.getenv('AWS_ENDPOINT_URL')
NORMALIZE: bool = env.getenv('NORMALIZE', 'False').lower() in ('true', '1', 't', 'y', 'yes')

# Supported models
SUPPORTED_MODELS = [
    "amazon.titan-embed-text-v1",
    "amazon.titan-embed-g1-text-02"
]


def api_config_schema():
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "The model to be used. Must be one of the supported models.",
                "enum": SUPPORTED_MODELS,
                "default": MODEL
            },
            "region_name": {
                "type": "string",
                "description": "The AWS region name.",
                "default": REGION_NAME
            },
            "credentials_profile_name": {
                "type": "string",
                "description": "The AWS credentials profile name.",
                "default": CREDENTIALS_PROFILE_NAME
            },
            "endpoint_url": {
                "type": "string",
                "description": "The AWS endpoint URL.",
                "default": ENDPOINT_URL
            },
            "normalize": {
                "type": "boolean",
                "description": "Whether to normalize the embeddings.",
                "default": NORMALIZE
            }
        },
        "required": ["model"],
        "additionalProperties": False
    }
