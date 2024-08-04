import env_setup as env

# Configuration settings with default values
DIMENSIONS = int(env.getenv('COHERE_DIMENSION'))
MODEL: str = "embed-english-v3.0"  # Default model

SUPPORTED_INPUT_TYPE = ["search_query", "search_document", "classification", "clustering"]
INPUT_TYPE = "search_query"

SUPPORTED_ENCODING_FORMAT = ["float", "int8", "uint8", "binary", "ubinary"]
ENCODING_FORMAT = "float"
API_KEY: str = env.getenv('COHERE_API_KEY')

# Supported models
SUPPORTED_MODELS = [
    "embed-english-v3.0",
    "embed-multilingual-v3.0"
    "embed-english-light-v3.0"
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
            },
            "input_type": {
                "type": "string",
                "description": "The type of input for the service.",
                "enum": SUPPORTED_INPUT_TYPE,
                "default": INPUT_TYPE
            }
        },
        "required": ["api_key"],
        "additionalProperties": False
    }
