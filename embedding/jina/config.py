import json

import env_setup as env

# Configuration settings with default values
DIMENSIONS = int(env.getenv('JINA_DIMENSIONS'))  # Default dimensions, adjust based on the Jina model used
MODEL: str = "jina-clip-v1"  # Default model
SUPPORTED_ENCODING_FORMAT = ["float", "base64"]
ENCODING_FORMAT = "float"

API_KEY: str = env.getenv('JINA_API_KEY')

# Supported models
SUPPORTED_MODELS = [
    "jina-clip-v1",
    # Add other supported models here if needed
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