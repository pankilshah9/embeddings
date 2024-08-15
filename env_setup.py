from dotenv import load_dotenv
import os
from typing import Any


class Setup:
    env_loaded = False

    @staticmethod
    def load_env():
        dotenv_path = os.path.join(os.path.dirname(__file__), 'environment.env')
        load_dotenv(dotenv_path=dotenv_path)

    @staticmethod
    def getenv(key, default):
        if not Setup.env_loaded:
            Setup.env_loaded = True
            Setup.load_env()
        return os.getenv(key) or default


def getenv(key: str, default: Any = None):
    return Setup.getenv(key, default)
