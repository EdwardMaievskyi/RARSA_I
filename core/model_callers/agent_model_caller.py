import logging
import json
import uuid
from typing import Dict, List, Tuple, Any, Callable
import time

from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether
from langchain_core.messages import AIMessage, SystemMessage
from retry import retry

from config import LLMConfig
from core.prompts import MAIN_SYSTEM_PROMPT
from core.state_models import AgentState
from core.tools.openai_tools_schema import openai_tools_schemas
from core.tools.anthropic_tools_schema import anthropic_tools_schemas
from core.tools.gemini_tools_schemas import gemini_tools_schemas


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResponseMessage:
    """Hybrid message class that supports both dictionary and attribute access.
    """
    def __init__(self, role: str, content: str, tool_calls: List = None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)


class ToolCallObject:
    """Tool call object that mimics OpenAI's structure."""
    def __init__(self, id: str, tool_type: str,
                 function_name: str, function_arguments: str):
        self.id = id
        self.type = tool_type
        self.function = self.Function(function_name, function_arguments)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    class Function:
        def __init__(self, name: str, arguments: str):
            self.name = name
            self.arguments = arguments

        def __getitem__(self, key):
            return getattr(self, key)

        def get(self, key, default=None):
            return getattr(self, key, default)


class AgentModelCaller:
    """A unified provider class to interact with multiple LLM SDKs."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        provider = config.preferred_ai_model_provider
        if provider == "openai" and config.openai_api_key:
            self.openai_model_name = \
                self.config.openai_cost_saving_model_name
            self.openai_client = OpenAI(api_key=config.openai_api_key)
        elif provider == "anthropic" and config.anthropic_api_key:
            self.anthropic_model_name = \
                self.config.anthropic_cost_saving_model_name
            anthropic_model = ChatAnthropic(
                model=self.anthropic_model_name,
                api_key=config.anthropic_api_key,
                temperature=0.4,
                max_tokens=8192
            )
            self.anthropic_model_armed = \
                anthropic_model.bind_tools(anthropic_tools_schemas)
        elif provider == "google" and config.google_api_key:
            self.gemini_model_name = \
                self.config.gemini_cost_saving_model_name
            gemini_model = ChatGoogleGenerativeAI(
                model=self.gemini_model_name,
                api_key=config.google_api_key,
                temperature=0.4,
                max_retries=self.config.max_retry_attempts
            )
            self.gemini_model_armed = \
                gemini_model.bind_tools(gemini_tools_schemas)
        elif config.together_api_key and \
                config.preferred_ai_model_provider == "together":
            self.together_model_name = \
                self.config.together_cost_saving_model_name
            together_model = ChatTogether(
                model=self.together_model_name,
                api_key=config.together_api_key,
                temperature=0.2,
                max_tokens=8192
            )
            self.together_model_armed = \
                together_model.bind_tools(openai_tools_schemas)
        else:
            raise ValueError(
                "Incorrect model provider value or no API key provided. "
                f"Provider: '{provider}'. "
                "Supported providers: 'openai', 'anthropic', 'google', "
                "'together'."
            )

    @retry(
        exceptions=Exception,
        tries=3,
        delay=1,
        backoff=2,
        max_delay=10,
        logger=logger
    )
    def _call_openai(self, messages: List[Dict]):
        """Handles the API call to OpenAI."""
        self.logger.info(f"Calling OpenAI model: {self.openai_model_name}")

        response = self.openai_client.chat.completions.create(
            model=self.openai_model_name,
            messages=messages,
            tools=openai_tools_schemas,
            tool_choice="auto",
        )
        return response.choices[0].message

    @retry(
        exceptions=Exception,
        tries=3,
        delay=1,
        backoff=2,
        max_delay=10,
        logger=logger
    )
    def _call_anthropic(self, messages: List[Dict]):
        """Handles the API call to Anthropic using LangChain."""
        self.logger.info(
            f"Calling Anthropic model: {self.anthropic_model_name}"
        )
        langchain_messages = self._convert_to_langchain_messages(messages)
        return self.anthropic_model_armed.invoke(langchain_messages)

    @retry(
        exceptions=Exception,
        tries=3,
        delay=1,
        backoff=2,
        max_delay=10,
        logger=logger
    )
    def _call_google(self, messages: List[Dict]) -> AIMessage:
        """Handles the API call to Google using LangChain."""
        self.logger.info(f"Calling Google model: {self.gemini_model_name}")
        langchain_messages = self._convert_to_langchain_messages(messages)
        return self.gemini_model_armed.invoke(langchain_messages)

    @retry(
        exceptions=Exception,
        tries=3,
        delay=1,
        backoff=2,
        max_delay=10,
        logger=logger
    )
    def _call_together(self, messages: List[Dict]) -> AIMessage:
        """Handles the API call to Together.ai using LangChain."""
        self.logger.info(
            f"Calling Together.ai model: {self.together_model_name}"
        )
        langchain_messages = self._convert_to_langchain_messages(messages)
        return self.together_model_armed.invoke(langchain_messages)

    def _validate_tool_call(self, tc) -> bool:
        """Validates a tool call object has required fields."""
        if not tc:
            return False

        if hasattr(tc, 'function'):
            return all([
                hasattr(tc, 'id'),
                hasattr(tc.function, 'name'),
                hasattr(tc.function, 'arguments')
            ])
        elif hasattr(tc, 'id'):
            return all([
                hasattr(tc, 'name'),
                hasattr(tc, 'args')
            ])
        else:
            return all([
                tc.get('id'),
                tc.get('name'),
                tc.get('args') is not None
            ])

    def _adapt_model_output(
        self, response_obj: AIMessage
    ) -> ResponseMessage:
        """Adapts model outputs to match OpenAI's format."""
        tool_calls = []
        if hasattr(response_obj, 'tool_calls') and response_obj.tool_calls:
            for tc in response_obj.tool_calls:
                if not self._validate_tool_call(tc):
                    self.logger.warning(
                        f"Invalid tool call: {tc}"
                    )
                    continue

                if hasattr(tc, 'function'):
                    tc_id = tc.id
                    name = tc.function.name
                    args = tc.function.argument
                elif hasattr(tc, 'id'):
                    tc_id = tc.id
                    name = tc.name
                    args = tc.args
                else:
                    tc_id = tc.get('id', f"call_{uuid.uuid4()}")
                    name = tc.get('name', '')
                    args = tc.get('args', {})

                try:
                    args = json.loads(args) if isinstance(args, str) else args
                    if not isinstance(args, dict):
                        self.logger.warning(f"Invalid args type: {type(args)}")
                        args = {}
                except Exception as e:
                    self.logger.warning(f"Error parsing arguments: {e}")
                    args = {}

                if name == 'ResearchSummary':
                    if 'sources' in args and isinstance(args['sources'], str):
                        try:
                            sources = json.loads(args['sources'])
                            if isinstance(sources, list):
                                args['sources'] = sources
                        except json.JSONDecodeError:
                            self.logger.warning("Failed to parse sources")
                            args['sources'] = []

                tool_calls.append(ToolCallObject(
                    id=tc_id,
                    tool_type="function",
                    function_name=name,
                    function_arguments=json.dumps(args)
                ))

        return ResponseMessage(
            role="assistant",
            content=response_obj.content,
            tool_calls=tool_calls
        )

    def _convert_to_langchain_messages(self, messages: List[Dict]) -> List:
        """Converts messages to LangChain format."""
        langchain_messages = []
        for msg in messages:
            if hasattr(msg, 'role'):
                role = msg.role
                content = msg.content or ""
                tool_calls = getattr(msg, 'tool_calls', [])
            else:
                role = msg.get("role")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                from langchain_core.messages import HumanMessage
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                if tool_calls:
                    tool_calls_data = self._process_tool_calls(tool_calls)
                    langchain_messages.append(AIMessage(
                        content=content, 
                        tool_calls=tool_calls_data
                    ))
                else:
                    langchain_messages.append(AIMessage(content=content))
            elif role == 'tool':
                from langchain_core.messages import ToolMessage
                tool_call_id =\
                    msg.get('tool_call_id',
                            'unknown') if isinstance(msg, dict) else 'unknown'
                langchain_messages.append(ToolMessage(
                    content=content, 
                    tool_call_id=tool_call_id
                ))

        if (not langchain_messages or 
                langchain_messages[0].type != "system"):
            langchain_messages.insert(
                0, 
                SystemMessage(content=MAIN_SYSTEM_PROMPT)
            )
        return langchain_messages

    def _process_tool_calls(self, tool_calls: List) -> List[Dict]:
        """Processes tool calls into a standardized format."""
        tool_calls_data = []
        for tc in tool_calls:
            if not self._validate_tool_call(tc):
                self.logger.warning(f"Invalid tool call: {tc}")
                continue

            tc_id, name, args = self._extract_tool_call_data(tc)
            args = self._parse_and_validate_args(args, name)

            tool_calls_data.append({
                "name": name,
                "args": args,
                "id": tc_id
            })
        return tool_calls_data

    def _extract_tool_call_data(self, tc) -> Tuple[str, str, Any]:
        """Extracts tool call data from various formats."""
        if hasattr(tc, 'function'):
            return tc.id, tc.function.name, tc.function.arguments
        elif hasattr(tc, 'id'):
            return tc.id, tc.name, tc.args
        else:
            return (
                tc.get('id', f"call_{uuid.uuid4()}"),
                tc.get('name', ''),
                tc.get('args', {})
            )

    def _parse_and_validate_args(self, args: Any, name: str) -> Dict:
        """Parses and validates tool call arguments."""
        try:
            args = json.loads(args) if isinstance(args, str) else args
            if not isinstance(args, dict):
                self.logger.warning(f"Invalid args type: {type(args)}")
                return {}
        except Exception as e:
            self.logger.warning(f"Error parsing arguments: {e}")
            return {}

        if name == 'ResearchSummary':
            args = self._handle_research_summary_sources(args)

        return args

    def _handle_research_summary_sources(self, args: Dict) -> Dict:
        """Handles special case for ResearchSummary sources."""
        if 'sources' in args and isinstance(args['sources'], str):
            try:
                sources = json.loads(args['sources'])
                if isinstance(sources, list):
                    args['sources'] = sources
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse sources")
                args['sources'] = []
        return args

    def _call_with_retry(self,
                         provider: str,
                         call_func: Callable, *args, **kwargs):
        """Generic retry logic for API calls with exponential backoff."""
        last_error = None
        for attempt in range(self.config.max_retry_attempts):
            try:
                return call_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"{provider} API error on attempt {attempt + 1}: {e}"
                )
                if attempt < self.config.max_retry_attempts - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    self.logger.critical(
                        f"Max retry attempts reached for {provider} API"
                    )
                    raise last_error

    def call_model(self, state: AgentState) -> Dict:
        """Calls the appropriate model based on the configured provider."""
        messages = state['messages']
        provider = self.config.preferred_ai_model_provider
        error = None

        try:
            if provider == "openai":
                response_message = self._call_openai(messages)
            elif provider in ["anthropic", "google", "together"]:
                response_obj = getattr(self, f"_call_{provider}")(messages)
                response_message = self._adapt_model_output(response_obj)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            error = str(e)
            self.logger.error(f"Error calling {provider} API: {e}",
                              exc_info=True)
            error_content = (
                "An error occurred while processing your request. "
                f"Provider: {provider}, Error: {str(e)}. "
                "Please try again or rephrase your query."
            )
            response_message = ResponseMessage(
                role="assistant",
                content=error_content,
                tool_calls=[]
            )

        return {
            "messages": messages + [response_message],
            "current_iteration": state["current_iteration"] + 1,
            "error": error
        }
