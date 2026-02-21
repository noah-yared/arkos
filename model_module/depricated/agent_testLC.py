from langchain.prompts import ChatPromptTemplate

from langchain_core.tools import tool


from ArkModelRefactored import ArkModelLink
import yaml

# loads configuration for model
with open("../config_module/config.yaml", "r") as file:
    configuration = yaml.safe_load(file)

model_url = configuration["model_url"]


chat_model = ArkModelLink()


@tool
def multiply_two_numbers(number_1: int, number_2: int) -> str:
    """Multiplies two numbers."""

    return number_1 * number_2


# Bind the tools to the model
chat_model = chat_model.bind_tools([multiply_two_numbers])


# note: need to c

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a helpful AI assitant who gives funny answers to questions. You have tool calling functionality. If you recieve something that appears odd, it was a tool call you made. Give back the tool call and a response to it.",
        ),
        ("human", "{prompt}"),
    ]
)
chain = prompt_template | chat_model
result = chain.invoke({"prompt": "What is 51  * 90123 "})
print("******* \n\n **HERE**** \n\n")
print(result)
print("FINISHED")
