import json
import re
from typing import Dict, List, Union

import requests
from openai import OpenAI

from orion import logger
from orion.config.chatgpt_config import DeepSeekConfig, OpenAIConfig, LocalModelConfig

class ChatAPI:
    def __init__(
        self,
        config: Union[OpenAIConfig, DeepSeekConfig, LocalModelConfig ],
    ):
        self.messages: List[Dict[str, str]] = []
        self.history: List[Dict[str, str]] = []
        self.config = config
        
        if isinstance(config, DeepSeekConfig):
            self.client = None
            self.api_key = config.api_key
            self.api_url = "https://api.deepseek.com/v1/chat/completions"
        elif isinstance(config, LocalModelConfig):
            self.client = None
            self.api_key = config.api_key
            self.api_url = config.api_url  # This is the URL of my local model server
        elif isinstance(config, OpenAIConfig):
            self.client = OpenAI(api_key=config.api_key)
        else:
            print(type(config))
            raise ValueError("api_type can only be ['deepseek', 'openai']")

        self.model = config.model
        self.max_limit = min(config.limit - 2000, 32000)
        self.price = config.price
        self.usage_tokens = 0
        self.cost = 0  # USD money cost

    def add_function_message(self, content: str):
        if not re.search(r"^Function Return:", content):
            content = "Function Return:\n" + content
        self.messages.append({"role": "user", "content": content})
        self.history.append(self.messages[-1])

    def add_user_message(self, content: str):
        if not re.search(r"^User Utterance:", content):
            content = "User Utterance: " + content
        self.messages.append({"role": "user", "content": content})
        self.history.append(self.messages[-1])

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self.history.append(self.messages[-1])

    def get_system_response(self) -> str:
        try:
            self.do_truncation = False
            
            if isinstance(self.config, DeepSeekConfig):
                # DeepSeek API call using requests
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.model,
                    "messages": self.messages
                }
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()
                response_data = response.json()
                response_message = response_data["choices"][0]["message"]
                usage_tokens = response_data["usage"]["total_tokens"]
                message_content = response_message["content"]
            elif isinstance(self.config, LocalModelConfig):
                # Same as DeepSeek - just uses different api_url
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.model,
                    "messages": self.messages
                }
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()
                response_data = response.json()
                response_message = response_data["choices"][0]["message"]
                usage_tokens = response_data["usage"]["total_tokens"]
                message_content = response_message["content"]
            elif isinstance(self.config, OpenAIConfig):
                # OpenAI API call (existing code)
                response = self.client.chat.completions.create(
                    model=self.model, messages=self.messages
                )
                response_message = response.choices[0].message
                usage_tokens = response.usage.total_tokens
                message_content = response_message.content
            
            # Rest of the method stays the same...
            self.cost += usage_tokens * self.price / 1000
            logger.info(
                f"[DeepSeek] current model {self.model}, usage_tokens: {usage_tokens}, "
                f"cost: ${self.cost:.5f}, price: ${self.price:.5f}"
            )
            if usage_tokens > self.max_limit:
                logger.info(
                    f"[DeepSeek] truncate the conversation to avoid token usage limit, save money"
                )
                self.truncate()

            return message_content
        except Exception as e:
            logger.warning(f"[DeepSeek] Error: {e}")
            return "Sorry, I am not able to respond to that."

    def get_system_response_stream(self):
        if hasattr(self, 'api_url') and self.api_url:  # DeepSeek
            # DeepSeek streaming (if supported)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": self.messages,
                "stream": True
            }
            response = requests.post(self.api_url, headers=headers, json=data, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
        else:
            # OpenAI streaming (existing code)
            response = self.client.chat.completions.create(
                model=self.model, messages=self.messages, stream=True
            )
            for chuck in response:
                if len(chuck.choices) > 0 and chuck.choices[0].finish_reason != "stop":
                    if chuck.choices[0].delta.content is None:
                        continue
                    yield chuck.choices[0].delta.content

        # stream mode does not support token usage check, give a rough estimation
        usage_tokens = int(sum([len(item["content"]) for item in self.message]) / 3.5)
        self.usage_tokens = usage_tokens
        self.cost += usage_tokens * self.price / 1000
        logger.info(
            f"[DeepSeek] current model {self.model}, usage_tokens approximation: {usage_tokens},"
            f" cost: ${self.cost:.2f}, price: ${self.price:.2f}"
        )

        if usage_tokens > self.max_limit:
            logger.info(
                f"[DeepSeek] truncate the conversation to avoid token usage limit"
            )
            self.truncate()

    @property
    def message(self):
        return self.messages

    @message.setter
    def message(self, message):
        """
        Usually at the dialog begining
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_first_turn},
        {"role": "assistant", "content": assistant_message_first_turn},
        """
        self.init_length = len(message)
        self.messages = message
        self.history.extend(self.messages)

    def truncate(self, percentage: int = 3):
        self.do_truncation = True
        usr_idx = [
            idx
            for idx in range(len(self.messages))
            if self.messages[idx]["role"] == "user"
        ]
        middle_idx = usr_idx[len(usr_idx) // percentage]
        logger.info(
            f"\033[33m [ChatGPT] truncate the conversation at index: {middle_idx} from {usr_idx} \033[m"
        )
        self.messages = self.messages[: self.init_length] + self.messages[middle_idx:]

    def clear(self):
        """end the conversation"""
        self.messages = []
        self.history = []

    def clear_ctx(self):
        """clear the context"""
        usr_idx = [
            idx
            for idx in range(len(self.messages))
            if self.messages[idx]["role"] == "user"
        ]
        if len(usr_idx) == 0:
            return
        elif len(usr_idx) <= 3:
            last_idx = usr_idx[0]
        else:
            last_idx = usr_idx[-3]  # keep the last 4 user messages
        logger.info(
            f"\033[33m [ChatGPT] clear context, user message remove at index: {last_idx} from {usr_idx} \033[m"
        )
        self.messages = self.messages[: self.init_length] + self.messages[last_idx:]
