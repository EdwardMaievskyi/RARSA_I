import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

PRIMARY_MODEL_NAME = "o4-mini-2025-04-16"
