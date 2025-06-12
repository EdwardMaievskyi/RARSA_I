import logging
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Literal

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY", None)
PREFERRED_AI_MODEL_PROVIDER = os.getenv("PREFERRED_AI_MODEL_PROVIDER",
                                        "openai")
PRIMARY_OPENAI_MODEL_NAME = os.getenv("PRIMARY_OPENAI_MODEL_NAME",
                                      "o4-mini-2025-04-16")
COST_SAVING_OPENAI_MODEL_NAME = os.getenv("SECONDARY_OPENAI_MODEL_NAME",
                                          "o4-mini-2025-04-16")
PRIMARY_ANTHROPIC_MODEL_NAME = os.getenv("PRIMARY_ANTHROPIC_MODEL_NAME",
                                         "claude-3-7-sonnet-latest")
COST_SAVING_ANTHROPIC_MODEL_NAME = os.getenv("COST_SAVING_ANTHROPIC_MODEL_NAME",
                                             "claude-3-5-haiku-latest")
PRIMARY_GOOGLE_MODEL_NAME = os.getenv("PRIMARY_GOOGLE_MODEL_NAME",
                                      "gemini-2.5-flash-preview-05-20")
COST_SAVING_GOOGLE_MODEL_NAME = os.getenv("COST_SAVING_GOOGLE_MODEL_NAME",
                                          "gemini-2.5-flash-preview-05-20")
PRIMARY_TOGETHER_MODEL_NAME = os.getenv("PRIMARY_TOGETHERAI_MODEL_NAME",
                                        "meta-llama/llama-3.1-8b-instruct")
COST_SAVING_TOGETHER_MODEL_NAME = os.getenv("COST_SAVING_TOGETHERAI_MODEL_NAME",
                                            "meta-llama/llama-3.1-8b-instruct")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)
FRED_API_KEY = os.getenv("FRED_API_KEY", None)

MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", 3))


@dataclass
class LLMConfig:
    openai_api_key: Optional[str] = OPENAI_API_KEY
    anthropic_api_key: Optional[str] = ANTHROPIC_API_KEY
    google_api_key: Optional[str] = GOOGLE_API_KEY
    together_api_key: Optional[str] = TOGETHER_AI_API_KEY
    openai_primary_model_name: str = PRIMARY_OPENAI_MODEL_NAME
    anthropic_primary_model_name: str = PRIMARY_ANTHROPIC_MODEL_NAME
    gemini_primary_model_name: str = PRIMARY_GOOGLE_MODEL_NAME
    together_primary_model_name: str = PRIMARY_TOGETHER_MODEL_NAME
    openai_cost_saving_model_name: str = COST_SAVING_OPENAI_MODEL_NAME
    anthropic_cost_saving_model_name: str = COST_SAVING_ANTHROPIC_MODEL_NAME
    gemini_cost_saving_model_name: str = COST_SAVING_GOOGLE_MODEL_NAME
    together_cost_saving_model_name: str = COST_SAVING_TOGETHER_MODEL_NAME
    preferred_ai_model_provider: \
        Literal["openai",
                "anthropic",
                "google",
                "togetherai"] = PREFERRED_AI_MODEL_PROVIDER
    max_retry_attempts: int = MAX_RETRY_ATTEMPTS
