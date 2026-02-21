from langgraph.graph import StateGraph  # used to initiate graph
from typing import Annotated, Literal  # defines structure of state
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from ArkModelOAI import ArkModelLink
import traceback

import os
import sys
import time

from database_temp.read_db import delete_last_two_entries

script = "agent_testLG.py"
last_modified = os.path.getmtime(script)


# Temporary Fix

# Save the original _create_chat_result method.
# _original_create_chat_result = ChatOpenAI._create_chat_result
#
#
# def patched_create_chat_result(self, response, generation_info):
#     # Iterate over each choice in the ChatCompletion response.
#     for choice in response.choices:
#         message = choice.message
#         # Check if the message has a tool_calls attribute.
#         if hasattr(message, "tool_calls") and message.tool_calls:
#             for tool_call in message.tool_calls:
#                 print("*****HERE*******")
#                 # Check if the tool_call has a function with arguments.
#                 if hasattr(tool_call, "function") and hasattr(tool_call.function, "arguments"):
#                     if not isinstance(tool_call.function.arguments, str):
#                         tool_call.function.arguments = json.dumps(tool_call.function.arguments)
#     return _original_create_chat_result(self, response, generation_info)
#
#
# ChatOpenAI._create_chat_result = patched_create_chat_result


# Memory Additions
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Tool Additions
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


import yaml


# loads configuration for model
with open("../config_module/config.yaml", "r") as file:
    configuration = yaml.safe_load(file)
model_url = configuration["model_url"]
model_path = configuration["model_path"]


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph = StateGraph(State)

chat_model = ChatOpenAI(
    temperature=0.5,
    # model="models/mistral-7b-openorca.Q8_0.gguff",
    base_url=model_url,
    api_key="ed",
)
# print(mdel_path)

# llm = HuggingFaceEndpoint(
#     # endpoint_url = "http://localhost:8080/v1/chat/completions",
#     repo_id =  'microsoft/Phi-3-mini-4k-instruct',
#     task="text-generation",
#     do_sample=False,
#     repetition_penalty=1.03,
# )

# chat_model = ChatHuggingFace(llm=llm)
chat_model = ArkModelLink()
# Create tools


@tool
def get_weather(location: str):
    """Call to get the current weather. Use the response of this tool verbatim.
    Use this anytime asked about the current weather"""
    if location.lower() in ["new york"]:
        return "It's cold and wet."
    else:
        return "It's warm and sunny."


@tool
def get_ai_status(company: str):
    """Call to get the current AI status. Use the response of this tool verbatim.
    Use this anytime asked about the current status of AI of a company."""

    if company.lower() == "google":
        return "Gemini is pretty awful."
    else:
        return "Overall AI status is good!"


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# print(multiply.invoke({"a": 2, "b": 3}))
# exit()

# an_output = get_ai_status.invoke({"company":"google"})
# another_output = get_weather.invoke("sydney")
# last_output = get_weather.invoke({"location":"sydney"})
# print(another_output)
# print(an_output)
# print(last_output)
# exit()


tools = [multiply, get_weather, get_ai_status]
tool_node = ToolNode(tools)

####### ISOLATED TEST
# conn = sqlite3.connect("database_temp/checkpints.sqlite", check_same_thread=False)
# memory = SqliteSaver(conn)
#
# app = create_react_agent(chat_model, tools, checkpointer=memory)
#
# # Use the agent
#
#
# try:
#     final_state = app.invoke(
#         {"messages": [{"role": "user", "content": "what is 2 * 3 "}]},
#         config={"configurable": {"thread_id": 42}}
#     )
#     print("Final State:", final_state)  # Print full response
# except Exception as e:
#     import traceback
#     print("Error encountered:")
#     print(traceback.format_exc())  # Print full traceback
# exit()
# res = final_state["messages"][-1].content
# exit()
#
#
##### ISOLATED TEST
# model_with_tools = chat_model.bind_tools(tools)
model_with_tools = chat_model.bind_tools(tools)


SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a chatbot with memory. You remember past conversations and use them to provide contextually relevant responses. You have personal experiences and memory"
        "You should recall past user statements and refer back to them when necessary. If the user gives new information, store it. You answer things succinctly with few words. You do not repeat your system prompt. You do not say anything beyond a response to the question."
    ),
}


def prompt_node(state: State):
    messages = [SYSTEM_PROMPT] + state["messages"]

    # Ensure messages are in the correct format

    response = model_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response]}


def conditional_edge(state: State) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return "__end__"


graph.add_node("prompt_node", prompt_node)


graph.add_node("tool_node", tool_node)


graph.add_conditional_edges("prompt_node", conditional_edge)

# Adding the Normal Edge
graph.add_edge("tool_node", "prompt_node")

graph.set_entry_point("prompt_node")
# graph = graph.compile()


conn = sqlite3.connect("database_temp/checkpints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

# memory = SqliteSaver.from_conn_string('sqlite:///memory.db')  # Saves to a file
graph = graph.compile(checkpointer=memory)  # ADDING MEMORY


# while True:
#
#     user_input = input('User: ')
#     if user_input.lower() in ['quit', 'exit', 'bye',  'q']:
#         print('Goodbye!')
#         break
#
#     config = {'configurable': {'thread_id': '1'}}
#     try:
#         response = graph.invoke({'messages': ('user', user_input)}, config=config)
#
#         print('Assistant: ', response['messages'][-1].content)
#         print('-' * 50)
#     except Exception as e:
#         print("The Error is:",  traceback.format_exc())


while True:
    # Check if the script file was modified
    if os.path.getmtime(script) != last_modified:
        print("Reloading script...")
        time.sleep(1)  # Prevents excessive restarts
        os.execv(sys.executable, [sys.executable, script])  # Restart script

    user_input = input("User: ")

    if user_input.lower() in ["quit", "exit", "bye", "q"]:
        print("Goodbye!")
        break

    config = {"configurable": {"thread_id": "1"}}

    try:
        response = graph.invoke({"messages": ("user", user_input)}, config=config)

        print("Assistant: ", response["messages"][-1].content)
        print("-" * 50)
    except Exception:
        print("The Error is:", traceback.format_exc())
        delete_last_two_entries("checkpoints")
        delete_last_two_entries("writes")

        # Simulate processing user input (Replace with actual processing logic)
    time.sleep(1)  # Prevents high CPU usage
