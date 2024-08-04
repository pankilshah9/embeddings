import json

import env_setup as env

# Configuration settings with default values
DIMENSIONS = int(env.getenv('OPENAI_DIMENSIONS'))
MODEL: str = "text-embedding-3-small"  # Default model

SUPPORTED_ENCODING_FORMAT = ["float", "base64"]
ENCODING_FORMAT = "float"
API_KEY: str = env.getenv('OPENAI_API_KEY')

# Supported models
SUPPORTED_MODELS = [
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002"
]


def api_config_schema():
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "api_key": {
                "type": "string",
                "description": "The API key for accessing the service.",
                "default": API_KEY
            },
            "model": {
                "type": "string",
                "description": "The model to be used. Must be one of the supported models.",
                "enum": SUPPORTED_MODELS,
                "default": MODEL
            },
            "dimensions": {
                "type": "integer",
                "description": "Dimensions for the model, if applicable.",
                "default": DIMENSIONS
            },
            "encoding_format": {
                "type": "string",
                "description": "Encoding format for the data.",
                "enum": SUPPORTED_ENCODING_FORMAT,
                "default": ENCODING_FORMAT
            }
        },
        "required": ["api_key"],
        "additionalProperties": False
    }
