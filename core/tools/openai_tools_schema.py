from pydantic import BaseModel, Field

from core.tools.tools import (pydantic_to_openai_tool, wikipedia_search,
                              duckduckgo_search,
                              scrape_and_summarize_web_page, tavily_search,
                              WikipediaSearchArgs, DuckDuckGoSearchArgs,
                              ScrapeArgs, TavilySearchArgs)
from core.state_models import ResearchSummary


openai_tools_schemas = [
    pydantic_to_openai_tool(WikipediaSearchArgs,
                            "wikipedia_search",
                            wikipedia_search.__doc__),
    pydantic_to_openai_tool(DuckDuckGoSearchArgs,
                            "duckduckgo_search",
                            duckduckgo_search.__doc__),
    pydantic_to_openai_tool(ScrapeArgs,
                            "scrape_and_summarize_web_page",
                            scrape_and_summarize_web_page.__doc__),
    pydantic_to_openai_tool(TavilySearchArgs,
                            "tavily_search",
                            tavily_search.__doc__),
    pydantic_to_openai_tool(ResearchSummary,
                            "ResearchSummary",
                            "Use this function to provide the final research summary and its sources, or to indicate that no information was found."),
]
