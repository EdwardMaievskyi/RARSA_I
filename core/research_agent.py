import logging
from typing import Dict, Any, Callable, Optional

from core.graph_builder import search_agent_graph
from core.prompts import MAIN_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Research agent wrapper for handling queries and streaming responses."""

    def __init__(self):
        self.max_iterations = 20
        logger.debug("ResearchAgent initialized with " +
                     f"max_iterations={self.max_iterations}")

    def process_query(
            self,
            query: str,
            progress_callback: Optional[Callable[[str], None]] = None
            ) -> Dict[str, Any]:
        """Process a research query and return results."""
        logger.debug("process_query called with query length: " +
                     f"{len(query) if query else 0}")

        if not query or not query.strip():
            logger.warning("Empty or invalid query provided")
            return {
                "success": False,
                "error": "Please enter a valid query.",
                "summary": "",
                "sources": []
            }

        try:
            logger.info(f"Processing query: {query}")

            initial_messages = [
                {"role": "system", "content": MAIN_SYSTEM_PROMPT},
                {"role": "user", "content": query.strip()}
            ]

            initial_state = {
                "query": query.strip(),
                "messages": initial_messages,
                "final_answer": None,
                "max_iterations": self.max_iterations,
                "current_iteration": 0
            }

            final_event_data = None
            terminal_nodes = ["prepare_final_answer",
                              "force_no_info_iterations",
                              "force_no_info_no_tool"]

            logger.debug("Starting graph stream with " +
                         f"max_iterations={self.max_iterations}")

            for event_count, event in enumerate(search_agent_graph.stream(
                initial_state,
                {"recursion_limit": self.max_iterations * 2 + 5}
            )):
                logger.debug(f"Processing event {event_count + 1}, " +
                             f"current_node: {list(event.keys())[0]}")

                if progress_callback:
                    progress_callback(f"Processing step {event_count + 1}...")

                current_node = list(event.keys())[0]

                if current_node in terminal_nodes:
                    logger.debug(f"Reached terminal node: {current_node}")
                    final_event_data = event[current_node]
                    if 'final_answer' in final_event_data and \
                            final_event_data['final_answer'] is not None:
                        logger.info("Final answer found, " +
                                    "breaking from event loop")
                        break

            if final_event_data and final_event_data.get('final_answer'):
                final_summary = final_event_data['final_answer']
                sources_list = []

                if hasattr(final_summary, 'sources') and final_summary.sources:
                    logger.debug("Processing " +
                                 f"{len(final_summary.sources)} sources")
                    for source in final_summary.sources:
                        source_info = {
                            "title": getattr(source, 'title', 'Unknown Title'),
                            "source_name": getattr(source,
                                                   'source_name',
                                                   'Unknown Source'),
                            "url": getattr(source, 'url', None),
                            "snippet": (
                                getattr(source, 'snippet', '')[:300] + "..."
                                if getattr(source, 'snippet', '')
                                else "No snippet available"
                            ),
                        }
                        sources_list.append(source_info)

                logger.info("Query processed successfully " +
                            f"with {len(sources_list)} sources")
                return {
                    "success": True,
                    "summary": getattr(final_summary,
                                       'summary',
                                       'No summary available'),
                    "sources": sources_list,
                    "query": query.strip()
                }
            else:
                logger.warning("No final answer found in agent processing")
                return {
                    "success": False,
                    "error": "No results found or agent failed to " +
                    "complete the research.",
                    "summary": "",
                    "sources": []
                }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"An error occurred: {str(e)}",
                "summary": "",
                "sources": []
            }
