from pydantic import BaseModel, Field
from typing import List, Dict, Any, TypedDict, Optional, Annotated


class SearchResult(BaseModel):
    """A single, standardized search result from any source."""
    title: str = Field(description="The title of the search result.")
    url: str = Field(description="The URL of the search result.")
    snippet: str = Field(
        description="A brief summary or snippet of the content."
    )
    source_name: str = Field(
        description=(
            "The name of the source engine, e.g., 'Wikipedia', "
            "'DuckDuckGo', 'WebScraper', 'Tavily'."
        )
    )


class ResearchSummary(BaseModel):
    """The final, synthesized output of the research agent."""
    summary: str = Field(
        description=(
            "A comprehensive, synthesized answer to the user's query based "
            "on the search results. If no information is found, this field "
            "should explicitly state that."
        )
    )
    sources: List[SearchResult] = Field(
        description=(
            "A list of all the source materials used to generate the "
            "summary. Can be empty if no information was found."
        )
    )


class AgentState(TypedDict):
    """Represents the state of our search agent."""
    query: str
    messages: List[Dict[str, Any]]
    final_answer: Annotated[Optional[ResearchSummary],
                            "The final summary and sources."]
    max_iterations: int
    current_iteration: int
