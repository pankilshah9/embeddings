import env_setup as env
from typing import Literal

# Configuration settings with default values
DIMENSIONS = 1024
MODEL: str = "voyage-large-2-instruct"  # Default model4
SUPPORTED_INPUT_TYPE = Literal["query", "document", None]
INPUT_TYPE = None
TRUNCATION: bool = True
API_KEY: str = env.getenv('VOYAGE_API_KEY')

# Supported models
SUPPORTED_MODELS = [
    "voyage-large-2-instruct",
    "voyage-finance-2",
    "voyage-multilingual-2",
    "voyage-law-2",
    "voyage-code-2",
    "voyage-large-2",
    "voyage-2"
]


def api_config_schema():
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Embedding Configuration Schema",
        "type": "object",
        "properties": {
            "api_key": {
                "type": "string",
                "default": API_KEY
            },
            "dimensions": {
                "type": "integer",
                "default": DIMENSIONS
            },
            "model": {
                "type": "string",
                "enum": SUPPORTED_MODELS,
                "default": MODEL
            },
            "input_type": {
                "type": "string",
                "enum": SUPPORTED_INPUT_TYPE,
                "default": INPUT_TYPE
            },
            "truncation": {
                "type": "boolean",
                "default": TRUNCATION
            }

        },
        "required": ["api_key"],
        "additionalProperties": False
    }
