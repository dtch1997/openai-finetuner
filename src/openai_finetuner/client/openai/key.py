import os 

def get_active_key() -> str:
    """Get the active OpenAI API key from the environment."""
    return os.getenv("OPENAI_API_KEY")

def set_active_key(key: str) -> None:
    """Set the active OpenAI API key in the environment."""
    os.environ["OPENAI_API_KEY"] = key