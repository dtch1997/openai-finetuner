from openai import OpenAI
from openai_finetuner.system.key import KeyManager

key_manager = KeyManager()

class ClientManager:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._client = OpenAI(api_key=key_manager.get_key())
        return cls._instance

    @property
    def client(self) -> OpenAI:
        return self._client
