import os
from typing import Optional, Dict

def _read_env_keys() -> Dict[str, str]:
    keys = {}
    # Check main API key
    if main_key := os.getenv("OPENAI_API_KEY"):
        keys["default"] = main_key
        
    # Check numbered API keys
    for i in range(32):
        key_name = f"OPENAI_API_KEY_{i}"
        if key := os.getenv(key_name):
            keys[str(i).lower()] = key
            
    return keys

class KeyManager:
    """Manages multiple OpenAI API keys and facilitates switching between them."""
    def __init__(self):
        self._keys = _read_env_keys()
        self._active_key_name = next(iter(self._keys)) if self._keys else None

    def add_key(self, name: str, key: str):
        """Add a new API key with the given name."""
        self._keys[name] = key
        if self._active_key_name is None:
            self.set_active_key(name)

    def remove_key(self, name: str):
        """Remove the API key with the given name."""
        if name in self._keys:
            del self._keys[name]
            if self._active_key_name == name:
                self._active_key_name = next(iter(self._keys)) if self._keys else None

    def get_key(self, name: str | None = None) -> Optional[str]:
        """Get the API key with the given name."""
        if name is None:
            return self.get_active_key()
        return self._keys.get(name)

    def list_keys(self) -> list[str]:
        """List all stored key names."""
        return list(self._keys.keys())

    def set_active_key(self, name: str):
        """Set the active API key by name."""
        if name not in self._keys:
            raise ValueError(f"No key found with name: {name}")
        self._active_key_name = name

    def get_active_key(self) -> Optional[str]:
        """Get the currently active API key."""
        if self._active_key_name is None:
            return None
        return self._keys.get(self._active_key_name)

    def get_active_key_name(self) -> Optional[str]:
        """Get the name of the currently active API key."""
        return self._active_key_name
