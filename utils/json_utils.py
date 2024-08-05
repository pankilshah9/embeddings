import json
from typing import Dict, Any

from jsonschema import Draft7Validator


def validate_or_get_default_json(schema, json: Dict[str, Any]):
    if json is not None:
        validate_schema(schema, json)
    json = default_json_from_schema(schema, json)
    print(json)
    return json


def validate_schema(schema, json):
    validator = Draft7Validator(schema)
    errors = []

    # Collect all errors
    for error in validator.iter_errors(json):
        errors.append(f"{error.message}")

    if errors:
        raise ValueError(", ".join(errors))


def default_json_from_schema(schema, defaults=None):
    if defaults is None:
        defaults = {}

    def fill_defaults(properties):
        for key, value in properties.items():
            if key not in defaults:  # Check if the key does not already exist
                if 'default' in value:
                    defaults[key] = value['default']
                if 'properties' in value:
                    fill_defaults(value['properties'])

    fill_defaults(schema.get('properties', {}))

    # Add required fields with default values if they are not already in defaults
    for field in schema.get('required', []):
        if field not in defaults:
            defaults[field] = None

    return defaults
