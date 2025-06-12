import logging
from typing import Dict, List

from openai import OpenAI
import google.genai as genai
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_together import ChatTogether
import requests
from retry import retry

from config import LLMConfig, DEFAULT_MODEL_CONFIG
from core.prompts import SCRAPER_SYSTEM_PROMPT


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScrapperModelCaller:
    """A simplifier provider class to interact with multiple LLM SDKs."""

    def __init__(self, config: LLMConfig):
        """Initializes the clients for all supported LLM providers."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        if config.openai_api_key and \
                config.preferred_ai_model_provider == "openai":
            self.openai_client = OpenAI(api_key=config.openai_api_key)
        elif config.anthropic_api_key and \
                config.preferred_ai_model_provider == "anthropic":
            self.anthropic_model = ChatAnthropic(
                model=config.anthropic_cost_saving_model_name,
                api_key=config.anthropic_api_key,
                **DEFAULT_MODEL_CONFIG
            )
        elif config.google_api_key and \
                config.preferred_ai_model_provider == "google":
            self.gemini_client = genai.Client(
                api_key=config.google_api_key,
                http_options=genai.types.HttpOptions(
                    api_version='v1alpha'
                ),
                **DEFAULT_MODEL_CONFIG
            )
        elif config.together_api_key and \
                config.preferred_ai_model_provider == "together":
            self.together_model = ChatTogether(
                model=config.together_cost_saving_model_name,
                api_key=config.together_api_key,
                **DEFAULT_MODEL_CONFIG
            )
        else:
            raise ValueError(
                "Incorrect model provider value or no API key provided. " +
                f"Provider: '{config.preferred_ai_model_provider}'. " +
                "Supported providers: 'openai', 'anthropic', 'google', " +
                "'together'."
            )

    def _prepare_messages(self, messages: List[Dict]) -> List[Dict]:
        """Prepares messages for model calls."""
        user_messages = [msg for msg in messages if msg['role'] != 'system']
        return [SystemMessage(content=SCRAPER_SYSTEM_PROMPT)] + user_messages

    @retry(
        exceptions=requests.exceptions.RequestException,
        tries=3,
        delay=1,
        backoff=2,
        max_delay=10,
        logger=logger
    )
    def _call_openai_with_scraper(self, messages: List[Dict]) -> str:
        """Handles the API call to OpenAI with the scraper tool."""
        self.logger.info(
            "Calling OpenAI model: " +
            f"{self.config.openai_cost_saving_model_name}"
        )

        summary_response = self.openai_client.chat.completions.create(
            model=self.config.openai_cost_saving_model_name,
            messages=messages
        )
        return summary_response.choices[0].message.content

    @retry(
        exceptions=requests.exceptions.RequestException,
        tries=3,
        delay=1,
        backoff=2,
        max_delay=10,
        logger=logger
    )
    def _call_anthropic_with_scraper(self, messages: List[Dict]) -> str:
        """Handles the API call to Anthropic with the scraper tool."""
        self.logger.info(
            "Calling Anthropic model: " +
            f"{self.config.anthropic_cost_saving_model_name}"
        )
        prepared_messages = self._prepare_messages(messages)
        return self.anthropic_model.invoke(prepared_messages).content

    @retry(
        exceptions=requests.exceptions.RequestException,
        tries=3,
        delay=1,
        backoff=2,
        max_delay=10,
        logger=logger
    )
    def _call_google_with_scraper(self, messages: List[Dict]) -> str:
        """Handles the API call to Google for scraping purposes."""
        self.logger.info(
            "Calling Google model for scraping: " +
            f"{self.config.gemini_cost_saving_model_name}"
        )
        user_messages = [msg for msg in messages if msg['role'] != 'system']
        content = "\n\n".join([msg['content'] for msg in user_messages])

        response = self.gemini_client.models.generate_content(
            model=self.config.gemini_cost_saving_model_name,
            contents=content,
            config=genai.types.GenerateContentConfig(
                system_instruction=[SCRAPER_SYSTEM_PROMPT]
            )
        )
        return response.text

    @retry(
        exceptions=requests.exceptions.RequestException,
        tries=3,
        delay=1,
        backoff=2,
        max_delay=10,
        logger=logger
    )
    def _call_together_with_scraper(self, messages: List[Dict]) -> str:
        """Handles the API call to Together.ai for scraping purposes."""
        self.logger.info(
            "Calling Together.ai model for scraping: " +
            f"{self.config.together_cost_saving_model_name}"
        )
        prepared_messages = self._prepare_messages(messages)
        return self.together_model.invoke(prepared_messages).content

    def call_model_with_scraper(self, messages: List[Dict]) -> str:
        """Handles the API call to the model with the scraper tool."""
        self.logger.info(
            "Calling model with scraper tool: " +
            f"{self.config.preferred_ai_model_provider}"
        )

        if self.config.preferred_ai_model_provider == "openai":
            return self._call_openai_with_scraper(messages)
        elif self.config.preferred_ai_model_provider == "anthropic":
            return self._call_anthropic_with_scraper(messages)
        elif self.config.preferred_ai_model_provider == "google":
            return self._call_google_with_scraper(messages)
        elif self.config.preferred_ai_model_provider == "together":
            return self._call_together_with_scraper(messages)
