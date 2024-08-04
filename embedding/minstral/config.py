import json

import env_setup as env

# Configuration settings with default values
MODEL: str = "mistral-embed"  # Default model
API_KEY: str = env.getenv('MISTRAL_API_KEY')
DIMENSIONS = int(env.getenv('MISTRAL_DIMENSIONS', 1024))

# Supported models
SUPPORTED_MODELS = [
    "mistral-embed"
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
            }
        },
        "required": ["api_key"],
        "additionalProperties": False
    }
