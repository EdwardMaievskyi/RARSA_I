import logging
from typing import Any, Dict, List, Optional

import requests
import wikipedia
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from tavily import TavilyClient

from config import (TAVILY_API_KEY, LLMConfig, MAX_CHARS_FOR_SUMMARY,
                    DEFAULT_MAX_RESULTS, DEFAULT_MAX_SENTENCES, USER_AGENT)
from core.model_callers.scrapper_model_caller import ScrapperModelCaller
from core.prompts import SCRAPER_SYSTEM_PROMPT
from core.state_models import SearchResult


logger = logging.getLogger(__name__)

llm_config = LLMConfig()
llm_scraper = ScrapperModelCaller(llm_config)


def pydantic_to_openai_tool(
    pydantic_model: BaseModel,
    function_name: str,
    function_description: str
) -> Dict[str, Any]:
    """Converts Pydantic models to OpenAI function tool schema."""
    schema = pydantic_model.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": function_description,
            "parameters": {
                "type": "object",
                "properties": schema["properties"],
                "required": schema.get("required", []),
            },
        },
    }


def wikipedia_search(
    query: str,
    max_sentences: int = DEFAULT_MAX_SENTENCES
) -> List[SearchResult]:
    """
    Searches Wikipedia for the given query and returns a summary.
    Use this first for factual, encyclopedic queries.

    Args:
        query: The search query for Wikipedia.
        max_sentences: Maximum number of sentences for the summary.

    Returns:
        A list of SearchResult objects.
    """
    logger.info(f"TOOL: Executing Wikipedia search for: '{query}'")
    try:
        wikipedia.set_user_agent("MyResearchAgent/1.0 (myemail@example.com)")
        page = wikipedia.page(query, auto_suggest=True)
        summary = wikipedia.summary(
            query,
            sentences=max_sentences,
            auto_suggest=True
        )
        logger.debug(
            f"Wikipedia search successful for '{query}', "
            f"page title: {page.title}"
        )
        return [
            SearchResult(
                title=page.title,
                url=page.url,
                snippet=summary,
                source_name="Wikipedia"
            )
        ]
    except wikipedia.exceptions.PageError:
        logger.info(f"Wikipedia page for '{query}' not found")
        return [
            SearchResult(
                title=f"Page for '{query}' not found",
                url="",
                snippet=(
                    f"Wikipedia does not have a page specifically "
                    f"titled '{query}'."
                ),
                source_name="Wikipedia"
            )
        ]
    except wikipedia.exceptions.DisambiguationError as e:
        return _handle_wikipedia_disambiguation(query, e, max_sentences)
    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
        return []


def _handle_wikipedia_disambiguation(
    query: str,
    error: wikipedia.exceptions.DisambiguationError,
    max_sentences: int
) -> List[SearchResult]:
    """Handles Wikipedia disambiguation errors."""
    logger.info(
        f"Wikipedia search for '{query}' resulted in disambiguation. "
        f"Options: {error.options[:3]}"
    )

    if not error.options:
        return [
            SearchResult(
                title=f"Disambiguation error for '{query}'",
                url="",
                snippet=(
                    f"Wikipedia search for '{query}' led to a "
                    f"disambiguation page."
                ),
                source_name="Wikipedia"
            )
        ]

    try:
        summary = wikipedia.summary(error.options[0], sentences=max_sentences)
        page = wikipedia.page(error.options[0])
        logger.debug(
            f"Wikipedia disambiguation fallback successful for "
            f"'{error.options[0]}'"
        )
        return [
            SearchResult(
                title=page.title,
                url=page.url,
                snippet=(
                    f"Disambiguation for '{query}'. "
                    f"Showing results for '{error.options[0]}': {summary}"
                ),
                source_name="Wikipedia"
            )
        ]
    except Exception as e_inner:
        logger.error(
            f"Wikipedia disambiguation fallback failed for "
            f"'{error.options[0]}': {e_inner}"
        )
        options_str = (
            ', '.join(error.options[:3]) if error.options else 'None'
        )
        return [
            SearchResult(
                title=f"Disambiguation error for '{query}'",
                url="",
                snippet=(
                    f"Wikipedia search for '{query}' led to a disambiguation "
                    f"page. Example options: {options_str}."
                ),
                source_name="Wikipedia"
            )
        ]


def duckduckgo_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS
) -> List[SearchResult]:
    """
    Searches the web using DuckDuckGo for the given query.
    Use this for general web searches if Wikipedia is not sufficient or appropriate.
    It's fast and provides a good starting point for research.

    Args:
        query: The search query for DuckDuckGo.
        max_results: Maximum number of search results to return.

    Returns:
        A list of SearchResult objects.
    """
    logger.info(f"TOOL: Executing DuckDuckGo search for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                logger.warning(
                    f"DuckDuckGo search returned no results for: '{query}'"
                )
                return []

            logger.debug(
                f"DuckDuckGo search returned {len(results)} results for: "
                f"'{query}'"
            )
            return [
                SearchResult(
                    title=r.get("title", "No Title"),
                    url=r.get("href", ""),
                    snippet=r.get("body", "No Snippet"),
                    source_name="DuckDuckGo"
                ) for r in results
            ]
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}", exc_info=True)
        return []


def scrape_and_summarize_web_page(
    url: str,
    original_query: str
) -> List[SearchResult]:
    """
    Scrapes content from a given web page URL, then uses an LLM to analyze its 
    content in the context of the original_query and returns a concise summary.
    Use this when a URL from a previous search (e.g., DuckDuckGo) seems highly 
    relevant and needs deeper inspection than its snippet provides.

    Args:
        url: The URL of the web page to scrape and summarize.
        original_query: The original user research query to provide context.

    Returns:
        A list of SearchResult objects.
    """
    logger.info(
        f"TOOL: Executing Web Scraper and Summarizer for URL: '{url}' "
        f"regarding query: '{original_query}'"
    )

    try:
        content = _fetch_webpage_content(url)
        if not content:
            return _create_empty_scrape_result(url)

        text_content = _extract_text_content(content)
        if not text_content.strip():
            return _create_empty_content_result(url)

        summary = _generate_summary(text_content, original_query)
        page_title = _extract_page_title(content) or url

        return [
            SearchResult(
                title=page_title.strip(),
                url=url,
                snippet=summary.strip(),
                source_name="WebScraper"
            )
        ]
    except requests.RequestException as e:
        logger.error(f"WebScraper request failed for {url}: {e}")
        return _create_error_result(url, f"Error during web request: {e}")
    except Exception as e:
        logger.error(f"WebScraper general error for {url}: {e}", exc_info=True)
        return _create_error_result(url, f"An unexpected error occurred: {e}")


def _fetch_webpage_content(url: str) -> Optional[BeautifulSoup]:
    """Fetches and parses webpage content."""
    headers = {'User-Agent': USER_AGENT}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.content, 'html.parser')


def _extract_text_content(soup: BeautifulSoup) -> str:
    """Extracts text content from BeautifulSoup object."""
    paragraphs = soup.find_all('p')
    return "\n".join([p.get_text() for p in paragraphs])


def _extract_page_title(soup: BeautifulSoup) -> Optional[str]:
    """Extracts page title from BeautifulSoup object."""
    title_tag = soup.find('title')
    return title_tag.string if title_tag else None


def _generate_summary(text_content: str, original_query: str) -> str:
    """Generates summary using LLM."""
    text_content_for_summary = text_content[:MAX_CHARS_FOR_SUMMARY]
    logger.debug(
        f"Extracted {len(text_content)} characters, using "
        f"{len(text_content_for_summary)} for summary"
    )

    scraper_messages = [
        {"role": "system", "content": SCRAPER_SYSTEM_PROMPT},
        {"role": "user", "content": _create_summary_prompt(
            original_query,
            text_content_for_summary
        )}
    ]

    return llm_scraper.call_model_with_scraper(scraper_messages)


def _create_summary_prompt(query: str, content: str) -> str:
    """Creates the prompt for summary generation."""
    return (
        f"Original Research Query: '{query}'\n\n"
        f"Web Page Content (first {MAX_CHARS_FOR_SUMMARY} "
        f"characters):\n```\n{content}```\n\n"
        "Based *only* on the provided web page content and its relevance to "
        "the original research query, provide a concise summary snippet. If "
        "the page content does not seem relevant to the query, output "
        "'The page content does not appear to be relevant to the query.'"
    )


def _create_empty_scrape_result(url: str) -> List[SearchResult]:
    """Creates a result for failed scraping."""
    return [
        SearchResult(
            title=f"Content Scraped from {url}",
            url=url,
            snippet="No meaningful text content found on the page.",
            source_name="WebScraper"
        )
    ]


def _create_empty_content_result(url: str) -> List[SearchResult]:
    """Creates a result for empty content."""
    return [
        SearchResult(
            title=f"Content Scraped from {url}",
            url=url,
            snippet="No meaningful text content found on the page.",
            source_name="WebScraper"
        )
    ]


def _create_error_result(url: str, error_msg: str) -> List[SearchResult]:
    """Creates a result for error cases."""
    return [
        SearchResult(
            title=f"Error processing {url}",
            url=url,
            snippet=error_msg,
            source_name="WebScraper"
        )
    ]


def tavily_search(
    query: str,
    search_depth: str = "basic",
    max_results: int = DEFAULT_MAX_RESULTS
) -> List[SearchResult]:
    """
    Performs a web search using Tavily.
    Use this for in-depth research when Wikipedia, DuckDuckGo, and targeted web scraping
    do not yield enough information or when you need a quick summarized answer with sources.
    Tavily can provide high-quality, relevant, and detailed results.
    Set search_depth to "advanced" for more comprehensive results if basic is not enough.

    Args:
        query: The search query for Tavily.
        search_depth: Search depth ('basic' or 'advanced').
        max_results: Maximum number of search results.

    Returns:
        A list of SearchResult objects.
    """
    logger.info(
        f"TOOL: Executing Tavily search for: '{query}' "
        f"with depth '{search_depth}'"
    )

    if not TAVILY_API_KEY:
        logger.error("Tavily API key not found or not configured")
        return [
            SearchResult(
                title="Tavily API Key Error",
                url="",
                snippet=(
                    "Tavily API key is not configured. "
                    "Search cannot be performed."
                ),
                source_name="Tavily"
            )
        ]

    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results
        )
        results = response.get("results", [])
        logger.debug(
            f"Tavily search returned {len(results)} results for: '{query}'"
        )
        return [
            SearchResult(
                title=r.get("title", "No Title"),
                url=r.get("url", ""),
                snippet=(
                    r.get("content", r.get("raw_content", "No Snippet"))[:500]
                ),
                source_name="Tavily"
            ) for r in results
        ]
    except Exception as e:
        logger.error(f"Tavily search failed: {e}", exc_info=True)
        return []


class WikipediaSearchArgs(BaseModel):
    """Arguments for Wikipedia search."""
    query: str = Field(description="The search query for Wikipedia.")
    max_sentences: int = Field(
        default=DEFAULT_MAX_SENTENCES,
        description="Maximum number of sentences for the summary."
    )


class DuckDuckGoSearchArgs(BaseModel):
    """Arguments for DuckDuckGo search."""
    query: str = Field(description="The search query for DuckDuckGo.")
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        description="Maximum number of search results to return."
    )


class ScrapeArgs(BaseModel):
    """Arguments for web scraping."""
    url: str = Field(
        description=(
            "The URL of the web page to scrape and summarize."
        )
    )
    original_query: str = Field(
        description=(
            "The original user research query to provide context."
        )
    )


class TavilySearchArgs(BaseModel):
    """Arguments for Tavily search."""
    query: str = Field(description="The search query for Tavily.")
    search_depth: str = Field(
        default="basic",
        description=(
            "Search depth for Tavily ('basic' or 'advanced')."
        )
    )
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        description="Maximum number of search results."
    )


available_tools_map = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
    "scrape_and_summarize_web_page": scrape_and_summarize_web_page,
    "tavily_search": tavily_search,
}
