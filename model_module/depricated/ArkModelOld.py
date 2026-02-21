import pprint
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field
from openai import OpenAI

# Removed direct requests import as it's now handled by huggingface_hub internally for streaming
from huggingface_hub import AsyncInferenceClient  # New import for streaming

pp = pprint.PrettyPrinter()


# --- Custom Message Classes ---
# These classes define the structure for different types of messages
# in the conversation, replacing Langchain's BaseMessage, AIMessage, HumanMessage.
class Message(BaseModel):
    """Base class for all messages."""

    content: str
    role: str


class UserMessage(Message):
    """Represents a message from the user."""

    role: str = "user"


class AIMessage(Message):
    """
    Represents a message from the AI.
    Can include tool calls if the AI decides to use tools.
    """

    role: str = "assistant"
    # content is now Optional[str] to handle cases where the AI's turn is solely a tool call.
    content: Optional[str] = None
    # tool_calls stores the structured information about tool calls
    # as returned by the OpenAI API (list of dicts).
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ToolMessage(BaseModel):  # Changed to BaseModel as it now has tool_call_id
    """
    Represents the output of a tool execution.
    The tool_call_id links this message back to the specific tool call
    made by the AI.
    """

    role: str = "tool"
    tool_call_id: str
    content: str


# --- Custom Tool Class ---
# This class provides a standardized interface for defining tools
# that the language model can use, replacing Langchain's BaseTool.
class CustomTool(BaseModel):
    """
    A custom tool interface. Concrete tools should inherit from this
    and implement the 'invoke' method.
    """

    name: str = Field(description="The name of the tool.")
    description: str = Field(
        description="A description of the tool's purpose and how to use it."
    )
    args_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema for the tool's input arguments. "
        "This defines the parameters the tool expects.",
    )

    def invoke(self, args: Dict[str, Any]) -> Any:
        """
        Execute the tool with the given arguments.
        This method must be implemented by concrete tool classes.
        """
        raise NotImplementedError(
            "Invoke method must be implemented by concrete tool classes."
        )

    def to_openai_function_schema(self) -> Dict[str, Any]:
        """
        Converts the tool definition into the OpenAI function schema format.
        This format is used when describing available tools to the LLM.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema,
            },
        }


class ArkModelLink(BaseModel):
    """
    A custom chat model designed to interface with Hugging Face TGI
    servers that expose an OpenAI-compatible API, supporting tool calling.
    """

    model_name: str = Field(default="tgi")
    base_url: str = Field(default="http://localhost:8080/v1")
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.7)
    tools: Optional[List[CustomTool]] = Field(default_factory=list)

    def _convert_tools_to_openai_format(self) -> Optional[List[Dict[str, Any]]]:
        """
        Converts the list of internal CustomTool objects into the
        list of OpenAI function schemas required by the API.
        """
        if not self.tools:
            return None
        return [tool.to_openai_function_schema() for tool in self.tools]

    def _get_tool_by_name(self, name: str) -> Optional[CustomTool]:
        """
        Retrieves a CustomTool object from the internal list by its name.
        """
        return next((tool for tool in self.tools if tool.name == name), None)

    def make_llm_call(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Makes a call to the OpenAI-compatible LLM endpoint.

        Args:
            messages: A list of custom Message objects representing the conversation history.
            tools: An optional list of OpenAI function schemas to expose to the LLM.

        Returns:
            A dictionary containing:
            - 'tool_calls': A list of dictionaries representing requested tool calls (if any).
            - 'message': The content of the LLM's text response.
        """
        client = OpenAI(
            base_url=self.base_url,
            api_key="_",  # Placeholder API key. The TGI server doesn't usually require one.
        )

        # Convert custom Message objects into the format expected by the OpenAI API.
        openai_messages_payload = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                openai_messages_payload.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msg_dict = {"role": "assistant"}
                # Always include 'content' key for assistant messages.
                # If msg.content is None, set it to an empty string.
                msg_dict["content"] = msg.content if msg.content is not None else ""
                if msg.tool_calls:
                    # Tool calls from a previous AI turn need to be passed back to the API.
                    # They should already be in the correct dict format.
                    msg_dict["tool_calls"] = msg.tool_calls
                openai_messages_payload.append(msg_dict)
            elif isinstance(msg, ToolMessage):
                # For ToolMessage, 'tool_call_id' is required.
                openai_messages_payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            else:
                # Catch-all for generic Message objects or unknown types (shouldn't happen with strict typing)
                openai_messages_payload.append(
                    {"role": msg.role, "content": msg.content}
                )

        try:
            # Call the OpenAI API chat completions endpoint.
            chat_completion = client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages_payload,
                tools=tools,  # Pass available tools if provided
                tool_choice="auto",  # Allows the model to decide whether to use a tool
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            message_from_llm = chat_completion.choices[0].message

            # Convert LLM's tool_calls (which are Pydantic objects) to a list of dictionaries
            # for consistent internal handling and passing back into the API in future turns.
            tool_calls_as_dicts = []
            if message_from_llm.tool_calls:
                for tc in message_from_llm.tool_calls:
                    tool_calls_as_dicts.append(
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                # tc.function.arguments is already a dictionary from the openai client
                                "arguments": tc.function.arguments,
                            },
                            "type": "function",  # As per OpenAI's tool_calls specification
                        }
                    )

            return {
                "tool_calls": tool_calls_as_dicts if tool_calls_as_dicts else None,
                "message": message_from_llm.content,
            }
        except Exception as e:
            print(f"Error during LLM call: {e}")
            return {
                "tool_calls": None,
                "message": f"Error: An error occurred during LLM call: {e}",
            }

    def generate_response(self, initial_messages: List[Message]) -> AIMessage:
        """
        Generates a response from the model, potentially involving a two-step
        process if tool calls are requested.

        Args:
            initial_messages: The initial list of messages to start the conversation.

        Returns:
            An AIMessage object containing the final response content and any
            tool calls that were executed during the process.
        """
        # Create a mutable copy of the initial messages to build the conversation history.
        conversation_history = list(initial_messages)
        # Convert bound CustomTools into the OpenAI function schema format.
        tool_schemas = self._convert_tools_to_openai_format()

        # Step 1: Make initial LLM call to decide if a tool is needed or to provide a direct answer.
        response_step_1 = self.make_llm_call(conversation_history, tools=tool_schemas)
        llm_content_step_1 = response_step_1["message"]
        tool_calls_from_llm = response_step_1["tool_calls"]

        if tool_calls_from_llm:
            print("***** MODEL REQUESTED TOOLS ******")

            # Append the model's tool call message to the conversation history.
            # The content might be empty if the model's turn was solely to call a tool.
            conversation_history.append(
                AIMessage(content=llm_content_step_1, tool_calls=tool_calls_from_llm)
            )

            # Execute each requested tool and add its output to the conversation history.
            for tool_call_data in tool_calls_from_llm:
                tool_name = tool_call_data["function"]["name"]
                arguments = tool_call_data["function"]["arguments"]
                tool_call_id = tool_call_data["id"]

                tool = self._get_tool_by_name(tool_name)
                tool_output_str = ""
                if not tool:
                    # If tool is not found, log it and generate an error message
                    tool_output_str = f"Error: Tool '{tool_name}' was requested but not found. This functionality is not available."
                    print(f"DEBUG: {tool_output_str}")
                else:
                    try:
                        tool_output = tool.invoke(
                            arguments
                        )  # Execute the tool with the dictionary arguments
                        tool_output_str = str(
                            tool_output
                        )  # Convert tool output to string for message
                        print(
                            f"Tool '{tool_name}' invoked with args {arguments}. Result: {tool_output_str}"
                        )
                    except Exception as e:
                        tool_output_str = f"Error invoking tool '{tool_name}': {e}"
                        print(f"Error invoking tool '{tool_name}': {e}")

                # Append the tool's output as a ToolMessage to the conversation history.
                conversation_history.append(
                    ToolMessage(content=tool_output_str, tool_call_id=tool_call_id)
                )

                # Add a clarifying user-role prompt for the second LLM call,
                # informing it about the tool's execution status.
                arg_str = ", ".join([f"{k}={v}" for k, v in arguments.items()])
                if "Error:" in tool_output_str:
                    synthesis_prompt = (
                        f"You previously called the tool `{tool_name}` with arguments ({arg_str}), "
                        f"but it resulted in an error: {tool_output_str}. "
                        f"Please respond to the user's original question directly, acknowledging the tool error "
                        f"if relevant, or state that you cannot fulfill the request without the tool's functionality."
                    )
                else:
                    synthesis_prompt = (
                        f"You previously called the tool `{tool_name}` with arguments ({arg_str}). "
                        f"The tool returned the result: {tool_output_str}. "
                        f"Now, based on all the information provided (your previous response, the tool call, "
                        f"and its output), please answer the user's original question clearly and concisely."
                    )
                conversation_history.append(UserMessage(content=synthesis_prompt))

            # Step 2: Make the SECOND LLM call with the updated conversation history
            # (which now includes the executed tool calls and their outputs).
            # This call should not expose tools, as it's for synthesizing the final answer.
            final_response = self.make_llm_call(conversation_history, tools=None)
            final_content = final_response["message"]

            # The final AIMessage should report the tool calls that initiated this two-step process.
            # We explicitly ensure tool_calls are handled to prevent unexpected behavior in tests.
            # If the model requested a tool we don't have, we still report it in the final AIMessage,
            # but the content will reflect the handling of the unknown tool.
            return AIMessage(
                content=final_content,
                tool_calls=tool_calls_from_llm,  # Report the tools that were actually requested and processed
            )

        else:
            print("***** NO TOOL CALL USED ****")
            # If no tool was called in the first step, the LLM's initial response
            # is the direct answer.
            return AIMessage(content=llm_content_step_1, tool_calls=None)

    def bind_tools(self, tools: List[CustomTool]) -> "ArkModelLink":
        """
        Adds a list of CustomTool objects to the model instance,
        making them available for the LLM to use.
        """
        self.tools.extend(tools)
        return self

    async def astream_response(
        self,
        messages: List[Message],
    ) -> AsyncIterator[AIMessage]:
        """
        Asynchronously streams responses from the model using huggingface_hub.
        Yields AIMessage objects incrementally as chunks are received.
        """
        # Create an AsyncInferenceClient for streaming
        # The base_url might need to be adjusted based on your TGI setup if it's not http://localhost:8080
        hf_client = AsyncInferenceClient(
            base_url=self.base_url.replace("/v1", "")
        )  # Remove /v1 for hf_client

        # Convert custom Message objects into the format expected by the HF client
        hf_messages_payload = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                hf_messages_payload.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # HF client might be more lenient, but keeping content if present
                msg_dict = {"role": "assistant"}
                if msg.content is not None:
                    msg_dict["content"] = msg.content
                # Tool calls are generally not passed back into HF streaming directly for synthesis
                # if the tool execution is handled in the generate_response two-step process.
                # However, if you wanted the raw HF model to see tool calls in context for direct generation
                # (without the two-step invoke), you'd format them differently here.
                hf_messages_payload.append(msg_dict)
            elif isinstance(msg, ToolMessage):
                # HF client typically does not directly consume 'tool' role messages in chat completions.
                # If tool outputs need to be presented to the model in streaming, they would be integrated
                # into the 'assistant' or 'user' content, or a specific system message.
                # For this implementation, we assume tool execution happens outside this streaming method.
                pass  # Tool messages are not typically part of the direct streaming input for HF models.
            else:
                hf_messages_payload.append({"role": msg.role, "content": msg.content})

        try:
            stream = await hf_client.chat.completions.create(
                messages=hf_messages_payload,
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,  # Crucial for streaming responses
                # Tools are not directly passed to the streaming endpoint this way in HF
                # If tool functions were to be used by the model itself, it would be configured in the model.
                # Here, we're assuming tool calls are part of the two-step generate_response process.
            )

            # Corrected syntax: `async for chunk in stream:`
            async for chunk in stream:
                # `chunk.choices[0].delta.content` holds the streamed text
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    yield AIMessage(content=content_chunk)
                # If tool calls were streamed (which is less common for simple text gen),
                # you would process `chunk.choices[0].delta.tool_calls` here.
                # For this setup, we're handling tool calls in `generate_response`.

        except Exception as e:
            print(f"Error during streaming request with huggingface_hub: {e}")
            yield AIMessage(
                content=f"Error: Could not connect to stream or stream error: {e}"
            )
            return


# --- Example Usage ---
# To run this example:
# 1. Install required libraries:
#    pip install openai pydantic huggingface_hub requests pytz
# 2. Start a Hugging Face Text Generation Inference (TGI) server with OpenAI compatibility.
#    Example Docker command:
#    docker run --rm -it -p 8080:80 -v ~/.cache/huggingface:/data \
#        ghcr.io/huggingface/text-generation-inference:latest \
#        --model-id bigcode/starcoder2-3b --port 80 --enable-openai-api
#    (Replace 'bigcode/starcoder2-3b' with your desired model.)
# 3. Run this Python script.

if __name__ == "__main__":
    print("Initializing ArkModelLink...")
    # Instantiate the model, assuming TGI server is at http://localhost:8080/v1
    chat_model = ArkModelLink(base_url="http://localhost:8080/v1")

    # --- Define Dummy Tools ---
    # These are example tools that the LLM can decide to call.
    class GetCurrentWeatherTool(CustomTool):
        name: str = "get_current_weather"
        description: str = "Get the current weather in a given location."
        args_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "fahrenheit",
                },
            },
            "required": ["location"],
        }

        def invoke(self, args: Dict[str, Any]) -> str:
            location = args.get("location")
            unit = args.get("unit", "fahrenheit")
            print(
                f"DEBUG: Invoking get_current_weather for '{location}' in unit '{unit}'"
            )
            # Simulate an external API call to get weather data
            if "San Francisco, CA" == location:
                return f"The current weather in San Francisco, CA is 72 degrees {unit} and sunny."
            elif "New York, NY" == location:
                return f"The current weather in New York, NY is 60 degrees {unit} and cloudy with a chance of rain."
            else:
                return f"Weather data not available for '{location}'."

    class GetCurrentTimeTool(CustomTool):
        name: str = "get_current_time"
        description: str = "Get the current time in a specified timezone."
        args_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to get the time for, e.g., 'America/Los_Angeles'",
                }
            },
            "required": ["timezone"],
        }

        def invoke(self, args: Dict[str, Any]) -> str:
            import datetime
            import pytz  # Requires 'pytz' library: pip install pytz

            timezone_str = args.get("timezone", "UTC")
            try:
                tz = pytz.timezone(timezone_str)
                now = datetime.datetime.now(tz)
                return (
                    f"The current time in {timezone_str} is {now.strftime('%H:%M:%S')}."
                )
            except pytz.UnknownTimeZoneError:
                return f"Error: Unknown timezone '{timezone_str}'."
            except Exception as e:
                return f"Error getting time: {e}"

    # Bind the dummy tools to the chat model.
    print("Binding dummy tools...")
    chat_model.bind_tools(tools=[GetCurrentWeatherTool(), GetCurrentTimeTool()])

    # --- Test Case 1: Question that triggers the weather tool ---
    print("\n--- Test 1: Tool call (Weather) ---")
    messages_with_weather_tool_call = [
        UserMessage(content="What's the weather like in San Francisco, CA?"),
    ]
    response_with_weather_tool = chat_model.generate_response(
        messages_with_weather_tool_call
    )
    print("\nResponse (with weather tool call):")
    pp.pprint(response_with_weather_tool.model_dump())  # Changed to model_dump()
    assert "72 degrees" in response_with_weather_tool.content
    assert response_with_weather_tool.tool_calls is not None
    assert (
        response_with_weather_tool.tool_calls[0]["function"]["name"]
        == "get_current_weather"
    )

    # --- Test Case 2: Question that triggers the time tool ---
    print("\n--- Test 2: Tool call (Time) ---")
    messages_with_time_tool_call = [
        UserMessage(content="What time is it in America/New_York?"),
    ]
    response_with_time_tool = chat_model.generate_response(messages_with_time_tool_call)
    print("\nResponse (with time tool call):")
    pp.pprint(response_with_time_tool.model_dump())  # Changed to model_dump()
    assert "The current time in America/New_York" in response_with_time_tool.content
    assert response_with_time_tool.tool_calls is not None
    assert (
        response_with_time_tool.tool_calls[0]["function"]["name"] == "get_current_time"
    )

    # --- Test Case 3: Question that does NOT trigger a tool (direct answer) ---
    print("\n--- Test 3: No tool call (direct answer) ---")
    messages_no_tool_call = [
        UserMessage(content="Tell me a fun fact about cats."),
    ]
    response_no_tool = chat_model.generate_response(messages_no_tool_call)
    print("\nResponse (no tool call):")
    pp.pprint(response_no_tool.model_dump())  # Changed to model_dump()
    assert response_no_tool.content is not None and len(response_no_tool.content) > 0

    # --- Test Case 4: Asynchronous Streaming Response ---
    print("\n--- Test 4: Asynchronous streaming response ---")

    async def run_streaming_test():
        stream_messages = [
            UserMessage(
                content="Tell me a short story about a brave knight and a dragon."
            )
        ]
        print("\nStreaming response:")
        full_content = ""
        async for chunk in chat_model.astream_response(stream_messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)  # Print content as it streams
                full_content += chunk.content
            if chunk.tool_calls:
                # In a real application, you would accumulate these tool_calls deltas
                # to reconstruct the full tool call object. For this example, we just print.
                print(f"\n[DEBUG: Streamed tool_calls delta: {chunk.tool_calls}]")
        print("\n--- Streaming test finished ---")
        print(f"Full streamed content length: {len(full_content)}")

    import asyncio

    try:
        # Run the asynchronous streaming test.
        asyncio.run(run_streaming_test())
    except RuntimeError as e:
        if "cannot run the event loop while another loop is running" in str(e):
            print(
                "\nNote: Asyncio runtime error detected. This usually happens when running multiple "
                "async functions sequentially in a single script without proper management. "
                "To resolve, execute this test in a fresh Python interpreter session or restructure "
                "your test suite."
            )
        else:
            raise

    print("\nAll refactoring and tests complete.")
    print(
        "If you encountered 'Error: Could not connect to stream' or similar, ensure your "
        "TGI server is running at 'http://localhost:8080/v1' and is accessible."
    )
