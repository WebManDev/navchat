from attrs import define

class ChatGPTConfig:
    pass


####################### OpenAI #######################
class OpenAIConfig(ChatGPTConfig):
    api_type: str = "openai"
    api_key: str = "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"
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
    api_key: str = "sk-22d391f1036849cc9846287e55506a54"
    model: str = "deepseek-chat"  # or whatever DeepSeek's model name is
    limit: int = 32000
    price: float = 0.0001