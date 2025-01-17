import os
import re

from typing import Dict

DEFAULT_KEY_NAME = "default"

def _search_env_vars(regex: str):
    """Search environment variables by regular expression."""
    for key, value in os.environ.items():
        if re.search(regex, key):
            yield key, value

def _read_env_keys() -> Dict[str, str]:
    """Read API keys from environment variables."""
    keys = {}
    # Check main API key
    if main_key := os.getenv("OPENAI_API_KEY"):
        keys[DEFAULT_KEY_NAME] = main_key
        
    # Search for any API key starting with 'OPENAI_API_KEY_'
    regex = "OPENAI_API_KEY_.+"
    for key, value in _search_env_vars(regex):
        # Extract the key name from the environment variable name
        key_name = key.replace("OPENAI_API_KEY_", "").lower()
        keys[key_name] = value
            
    return keys

class KeyManager:
    """Manages API keys."""
    
    def __init__(self):
        self.keys = _read_env_keys()

    def get_key(self, key_name: str = DEFAULT_KEY_NAME) -> str:
        """Get an API key by name."""
        return self.keys.get(key_name, self.keys[DEFAULT_KEY_NAME])
    
    def list_keys(self) -> list[str]:
        """List all available API keys."""
        return list(self.keys.keys())

    def set_key(self, key_name: str = DEFAULT_KEY_NAME) -> None:
        """Set the API key to use."""
        os.environ["OPENAI_API_KEY"] = self.get_key(key_name)