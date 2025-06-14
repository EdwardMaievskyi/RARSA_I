import json
import logging
from typing import Dict, List, Union

from pydantic import ValidationError

from core.state_models import AgentState, SearchResult, ResearchSummary
from core.tools.tools import available_tools_map


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _create_tool_error_message(
    tool_call_id: str,
    tool_name: str,
    error_message: str
) -> Dict[str, str]:
    """Creates a standardized error message for tool execution failures."""
    return {
        "tool_call_id": tool_call_id,
        "role": "tool",
        "name": tool_name,
        "content": json.dumps([{"error": error_message}])
    }


def _validate_tool_output(
    tool_output: List[SearchResult],
    tool_name: str
) -> None:
    """Validates that tool output is a list of SearchResult objects."""
    if not isinstance(tool_output, list) or not all(
        isinstance(item, SearchResult) for item in tool_output
    ):
        raise ValueError(
            f"Tool {tool_name} did not return List[SearchResult]. "
            f"Returned: {type(tool_output)}"
        )


def _execute_tool(
    tool_name: str,
    tool_args: Dict,
    tool_call_id: str
) -> Dict[str, str]:
    """Executes a tool and returns its result message."""
    try:
        selected_tool_function = available_tools_map.get(tool_name)
        if not selected_tool_function:
            not_found_msg = (
                f"Tool '{tool_name}' not found in available_tools_map."
            )
            logger.error(not_found_msg)
            return _create_tool_error_message(
                tool_call_id,
                tool_name,
                not_found_msg
            )

        tool_output = selected_tool_function(**tool_args)
        _validate_tool_output(tool_output, tool_name)

        tool_output_serializable = [
            res.model_dump() for res in tool_output
        ]
        logger.debug(
            f"Tool {tool_name} executed successfully, "
            f"returned {len(tool_output)} results"
        )

        return {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "name": tool_name,
            "content": json.dumps(tool_output_serializable)
        }

    except json.JSONDecodeError as e:
        error_message = (
            f"Error decoding JSON arguments for tool {tool_name}: {e}. "
            f"Arguments received: '{tool_args}'"
        )
        logger.error(error_message)
        return _create_tool_error_message(
            tool_call_id,
            tool_name,
            error_message
        )

    except ValidationError as e:
        error_message = f"Argument validation error for tool {tool_name}: {e}"
        logger.error(error_message)
        return _create_tool_error_message(
            tool_call_id,
            tool_name,
            error_message
        )

    except Exception as e:
        error_message = f"Error executing tool {tool_name}: {e}"
        logger.error(error_message, exc_info=True)
        return _create_tool_error_message(
            tool_call_id,
            tool_name,
            error_message
        )


def call_tool(state: AgentState) -> AgentState:
    """The 'acting' node. Executes the tool chosen by the model."""
    logger.info("NODE: call_tool - Starting tool execution")
    last_message = state['messages'][-1]
    tool_messages_to_append = []

    if not last_message.tool_calls:
        logger.warning(
            "call_tool was called, but the last message had no tool_calls."
        )
        error_msg = {
            "role": "user",
            "content": (
                "Error: No tool was called. Please try again, ensuring you " +
                "select a tool or use ResearchSummary."
            )
        }
        return {"messages": state['messages'] + [error_msg]}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call.function.name
        tool_args_str = tool_call.function.arguments
        logger.info(
            f"EXECUTING: '{tool_name}' with args: {tool_args_str}"
        )

        try:
            tool_args = json.loads(tool_args_str)
            tool_message = _execute_tool(
                tool_name,
                tool_args,
                tool_call.id
            )
            tool_messages_to_append.append(tool_message)
        except json.JSONDecodeError as e:
            error_message = (
                f"Error decoding tool arguments: {e}. "
                f"Arguments received: '{tool_args_str}'"
            )
            logger.error(error_message)
            tool_messages_to_append.append(
                _create_tool_error_message(
                    tool_call.id,
                    tool_name,
                    error_message
                )
            )

    return {"messages": state['messages'] + tool_messages_to_append}


def _parse_sources_data(sources_data: Union[str, List]) -> List[SearchResult]:
    """Parses and validates sources data from various formats."""
    if isinstance(sources_data, str):
        try:
            sources_data = json.loads(sources_data)
            logger.info("Parsed sources from JSON string")
        except json.JSONDecodeError:
            logger.warning(
                "Could not parse sources string as JSON, " +
                "treating as empty list"
            )
            return []

    if not isinstance(sources_data, list):
        logger.warning(
            f"Sources data is not a list, got {type(sources_data)}, "
            "treating as empty list"
        )
        return []

    parsed_sources = []
    for s_data in sources_data:
        try:
            parsed_sources.append(SearchResult.model_validate(s_data))
        except ValidationError as e:
            logger.warning(
                f"Could not validate source data: {s_data}. Error: {e}"
            )
            parsed_sources.append(
                SearchResult(
                    title="Invalid Source Data",
                    url="",
                    snippet=str(s_data),
                    source_name="Error"
                )
            )

    return parsed_sources


def prepare_final_answer_node(state: AgentState) -> AgentState:
    """Extracts arguments from the ResearchSummary tool call and populates final_answer."""
    logger.info("NODE: prepare_final_answer_node - Preparing final answer")
    ai_message = state['messages'][-1]
    research_summary_call = next(
        (tc for tc in ai_message.tool_calls 
         if tc.function.name == "ResearchSummary"),
        None
    )

    if not research_summary_call:
        logger.error(
            "prepare_final_answer_node called without " +
            "ResearchSummary tool_call. Forcing generic no info."
        )
        return {
            "final_answer": ResearchSummary(
                summary=(
                    "Error: Agent attempted to finalize without " +
                    "proper ResearchSummary call."
                ),
                sources=[]
            )
        }

    try:
        args = json.loads(research_summary_call.function.arguments)
        summary_text = args.get("summary", "No summary provided by agent.")
        sources_data = args.get("sources", [])

        parsed_sources = _parse_sources_data(sources_data)
        final_data = ResearchSummary(
            summary=summary_text,
            sources=parsed_sources
        )

        logger.info(
            f"FINAL ANSWER PREPARED: {final_data.summary[:100]}... "
            f"({len(final_data.sources)} sources)"
        )
        return {"final_answer": final_data}

    except Exception as e:
        logger.error(
            f"Error in prepare_final_answer_node: {e}. Raw args: "
            f"{research_summary_call.function.arguments}",
            exc_info=True
        )
        return {
            "final_answer": ResearchSummary(
                summary=(
                    f"Error processing final answer: {e}. " +
                    "The web search might not have yielded information."
                ),
                sources=[]
            )
        }


def force_no_info_finish_node(
    state: AgentState,
    reason: str = (
        "LLM did not call ResearchSummary as instructed or " +
        "max iterations reached."
    )
) -> AgentState:
    """Provides a fallback final answer when the agent doesn't finish correctly."""
    logger.warning(f"NODE: force_no_info_finish_node. Reason: {reason}")
    last_message_content = (
        state['messages'][-1].get('content', '')
        if state['messages']
        else ''
    )

    if last_message_content and not state['messages'][-1].get('tool_calls'):
        summary_text = (
            "The agent provided a direct textual response " +
            "instead of using the ResearchSummary function: "
            f"'{last_message_content}'. No verifiable sources were "
            "cited through the structured process."
        )
    elif "max iterations reached" in reason.lower():
        summary_text = (
            "The research process was terminated due to " +
            "reaching the maximum iteration limit. The web search may " +
            "not have yielded a conclusive answer within the allowed steps."
        )
    else:
        summary_text = (
            "The web search did not give information to answer " +
            "the initial query, or the agent workflow concluded unexpectedly."
        )

    return {
        "final_answer": ResearchSummary(
            summary=summary_text,
            sources=[]
        )
    }


def should_continue(state: AgentState) -> str:
    """Router: Decides the next step."""
    logger.debug("ROUTER: should_continue - Determining next step")
    last_message = state['messages'][-1]

    if state["current_iteration"] >= state["max_iterations"]:
        logger.warning(
            f"Max iterations ({state['max_iterations']}) reached. " +
            "Forcing finish."
        )
        return "force_finish_due_to_iterations"

    if last_message.tool_calls:
        has_research_summary = any(
            tc.function.name == "ResearchSummary"
            for tc in last_message.tool_calls
        )
        if has_research_summary:
            logger.info(
                "'ResearchSummary' tool called. Preparing final answer."
            )
            return "prepare_final_answer"
        else:
            logger.debug(
                "Other tool call(s) detected. Continuing to 'action'."
            )
            return "continue_with_tool"
    else:
        logger.warning(
            "No tool calls from LLM. LLM might not have " +
            "followed instructions to call ResearchSummary. " +
            "Forcing finish."
        )
        return "force_finish_no_tool_call"
