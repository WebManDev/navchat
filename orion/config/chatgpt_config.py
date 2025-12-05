import os
from attrs import define

class ChatGPTConfig:
    pass


####################### OpenAI #######################
class OpenAIConfig(ChatGPTConfig):
    api_type: str = "openai"
    api_key: str = os.getenv("OPENAI_API_KEY", "")  # Set via environment variable
    model: str
    limit: int
    price: float


class OpenAIGPT35Config(OpenAIConfig):
    model: str = "gpt-3.5-turbo-0125"
    limit: int = 16000
    price: float = 0.0005


class OpenAIGPT4Config(OpenAIConfig):
    model: str = "gpt-4-turbo-preview"
    limit: int = 128000
    price: float = 0.01


####################### DeepSeek #######################
@define
class DeepSeekConfig:
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")  # Set via environment variable
    model: str = "deepseek-chat"  # or whatever DeepSeek's model name is
    limit: int = 32000
    price: float = 0.0001


####################### Local Model #######################

@define
class LocalModelConfig:
    api_key: str = "local"  # Not needed but keeping interface
    model: str = "local-model"
    limit: int = 32000
    price: float = 0.0  # Free!
    api_url: str = "http://localhost:8000/v1/chat/completions"  # Your Flask server