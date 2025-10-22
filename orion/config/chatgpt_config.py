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


####################### Azure #######################


class AzureConfig(ChatGPTConfig):
    api_type: str = "azure"
    api_key: str = "BgRXUeusbzVlVKMVQ05BEFkG5fH5HvLWsTiqF8DZVCwgFXa7TCIKJQQJ99BJACYeBjFXJ3w3AAABACOG0kpo"
    api_version: str = "2023-12-01-preview"
    azure_endpoint: str = "https://research1212.openai.azure.com/"
    model: str
    limit: int
    price: float


class AzureGPT35Config(AzureConfig):
    model: str = "gpt-35-turbo-16k-0613"
    limit: int = 16000
    price: float = 0.0005


class AzureGPT4Config(AzureConfig):
    model: str = "gpt-4-0125-preview"
    limit: int = 128000
    price: float = 0.01
