import json
import core.tools.tools as ctt
from langchain_core.tools import tool


@tool
def wikipedia_search(query: str, max_sentences: int = 3) -> str:
    """
    Searches Wikipedia for the given query and returns a summary.
    Use this first for factual, encyclopedic queries.

    Args:
        query: The search query for Wikipedia
        max_sentences: Maximum number of sentences for the Wikipedia summary (default: 3)

    Returns:
        JSON string containing list of SearchResult objects
    """
    results = ctt.wikipedia_search(query, max_sentences)
    return json.dumps([result.model_dump() for result in results])


@tool
def duckduckgo_search(query: str, max_results: int = 3) -> str:
    """
    Searches the web using DuckDuckGo for the given query.
    Use this for general web searches if Wikipedia is not sufficient or appropriate.

    Args:
        query: The search query for DuckDuckGo
        max_results: Maximum number of search results to return (default: 3)

    Returns:
        JSON string containing list of SearchResult objects
    """
    results = ctt.duckduckgo_search(query, max_results)
    return json.dumps([result.model_dump() for result in results])


@tool
def scrape_and_summarize_web_page(url: str, original_query: str) -> str:
    """
    Scrapes content from a given web page URL, then uses Claude to analyze its content
    in the context of the original_query and returns a concise summary.

    Args:
        url: The URL of the web page to scrape and summarize
        original_query: The original user research query to provide context for summarization

    Returns:
        JSON string containing list of SearchResult objects
    """
    results = ctt.scrape_and_summarize_web_page(url, original_query)
    return json.dumps([result.model_dump() for result in results])


@tool
def tavily_search(query: str, search_depth: str = "basic", max_results: int = 3) -> str:
    """
    Performs a web search using Tavily for in-depth research.

    Args:
        query: The search query for Tavily
        search_depth: Search depth for Tavily ('basic' or 'advanced')
        max_results: Maximum number of search results (default: 3)

    Returns:
        JSON string containing list of SearchResult objects
    """
    results = ctt.tavily_search(query, search_depth, max_results)
    return json.dumps([result.model_dump() for result in results])


anthropic_tools_schemas = [wikipedia_search,
                           duckduckgo_search,
                           scrape_and_summarize_web_page,
                           tavily_search]
