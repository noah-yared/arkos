import sys
import os

from typing import Optional, List
from pydantic import BaseModel, Field


from model_module.ArkModelNew import ArkModelLink, UserMessage, AIMessage, SystemMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from state_module.state import State


from state_module.state_registry import register_state



class ReasonedOutput(BaseModel):
    """
    Enforced reasoning contract for the agent state.
    No tools, no chain-of-thought.
    """
    intent: str = Field(..., description="What the agent is trying to accomplish")
    approach: List[str] = Field(..., description="High-level reasoning steps")
    needs_clarification: bool = Field(..., description="Whether more user input is required")
    clarifying_question: Optional[str] = Field(
        None, description="Single clarifying question if needed"
    )
    final: str = Field(..., description="User-facing response")


@register_state
class StateAI(State):
    type = "agent"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    async def run(self, context, agent):
        """
        Pure reasoning state.
        - One LLM call
        - Structured reasoning enforced via schema
        - No tools
        - No recovery heuristics
        """

        messages = context if isinstance(context, list) else context.get("messages", [])

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "reasoned_output",
                "schema": ReasonedOutput.model_json_schema(),
            },
        }

        system = SystemMessage(
            content=(
                "You are the agent reasoning state.\n"
                "No tools are available.\n"
                "Produce a JSON object matching the provided schema.\n"
                "Do not reveal chain-of-thought.\n"
                "Use concise, high-level reasoning steps only.\n"
            )
        )

        llm_context = [system] + messages
        output = await agent.call_llm(context=llm_context, json_schema=json_schema)
        print("Reasoned Output: \n\n", output )


        
        data = ReasonedOutput.model_validate_json(output.content)






        if data.needs_clarification:
            if not data.clarifying_question:
                raise ValueError("needs_clarification=True but no clarifying_question provided")
            # Include reasoning context with the clarifying question
            response = data.final + "\n\n" + data.clarifying_question if data.final else data.clarifying_question
            return AIMessage(content=response)

        return AIMessage(content=data.final)

