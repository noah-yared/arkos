"""Tests for format_tools_for_system_prompt from base_module/app.py.

The app module cannot be imported directly in tests because it initializes
the full agent stack (DB, LLM, etc.) at module level. We test the function
in isolation by defining it here (same logic as base_module/app.py:63-82).
"""

import pytest
from unittest.mock import MagicMock


def format_tools_for_system_prompt(tools: dict) -> str:
    """Copied from base_module/app.py for isolated testing."""
    lines = []
    lines.append("You have access to the following tools.")
    lines.append("Use them when appropriate. Only call tools that are listed below.")
    lines.append("")

    for name, tool in tools.items():
        lines.append(f"Tool name: {name}")
        if getattr(tool, "description", None):
            lines.append(f"Description: {tool.description}")
        if getattr(tool, "input_schema", None):
            lines.append("Input schema:")
            lines.append(str(tool.input_schema))
        lines.append("")

    return "\n".join(lines)


class TestFormatToolsForSystemPrompt:
    def test_empty_tools(self):
        result = format_tools_for_system_prompt({})
        assert "You have access to the following tools." in result
        assert "Use them when appropriate" in result

    def test_single_tool_name_only(self):
        tools = {"search": MagicMock(description=None, input_schema=None)}
        result = format_tools_for_system_prompt(tools)
        assert "Tool name: search" in result

    def test_tool_with_description(self):
        tool = MagicMock()
        tool.description = "Search the web for information"
        tool.input_schema = None
        result = format_tools_for_system_prompt({"web_search": tool})
        assert "Tool name: web_search" in result
        assert "Description: Search the web for information" in result

    def test_tool_with_input_schema(self):
        tool = MagicMock()
        tool.description = "Calculator"
        tool.input_schema = {"type": "object", "properties": {"expr": {"type": "string"}}}
        result = format_tools_for_system_prompt({"calc": tool})
        assert "Tool name: calc" in result
        assert "Input schema:" in result
        assert "expr" in result

    def test_multiple_tools(self):
        tools = {
            "search": MagicMock(description="Search", input_schema=None),
            "calc": MagicMock(description="Calculate", input_schema=None),
            "weather": MagicMock(description="Get weather", input_schema=None),
        }
        result = format_tools_for_system_prompt(tools)
        assert "Tool name: search" in result
        assert "Tool name: calc" in result
        assert "Tool name: weather" in result
        assert "Description: Search" in result
        assert "Description: Calculate" in result
        assert "Description: Get weather" in result

    def test_tool_without_description_attr(self):
        tool = object()
        result = format_tools_for_system_prompt({"plain_tool": tool})
        assert "Tool name: plain_tool" in result
        assert "Description:" not in result

    def test_output_starts_with_header(self):
        result = format_tools_for_system_prompt({})
        lines = result.strip().split("\n")
        assert lines[0] == "You have access to the following tools."

    def test_tools_separated_by_blank_lines(self):
        tools = {
            "a": MagicMock(description="Tool A", input_schema=None),
            "b": MagicMock(description="Tool B", input_schema=None),
        }
        result = format_tools_for_system_prompt(tools)
        # Each tool block ends with an empty line
        assert "\n\n" in result
