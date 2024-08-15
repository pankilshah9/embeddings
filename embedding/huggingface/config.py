import env_setup as env

# Configuration settings with default values
MODEL: str = "hkunlp/instructor-large"  # Default model
API_URL: str = env.getenv('HF_API_URL', None)
API_KEY: str = env.getenv('HF_API_READ_KEY')

# Supported models
SUPPORTED_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "hkunlp/instructor-large"
]


def api_config_schema():
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "api_url": {
                "type": "string",
                "description": "The custom inference endpoint url for accessing the service.",
                "default": API_URL
            },
            "api_key": {
                "type": "string",
                "description": "API key for the HuggingFace Inference API.",
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
