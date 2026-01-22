"""Tests for tool system."""
import pytest
from agent.tools.base import Tool, SubmitResult
from agent.tools.executor import ToolExecutor
from agent.core.content import ToolCall


class GetWeather(Tool):
    """Get weather for a location."""
    location: str

    async def run(self) -> str:
        return f"Weather in {self.location}: 72F"


class SlowTool(Tool):
    """A slow tool for timeout testing."""
    timeout = 0.1  # 100ms timeout

    async def run(self) -> str:
        import asyncio
        await asyncio.sleep(1)  # Sleep longer than timeout
        return "done"


class ConfirmationTool(Tool):
    """A tool that requires confirmation."""
    requires_confirmation = True
    action: str

    async def run(self) -> str:
        return f"Executed: {self.action}"


def test_tool_name_from_class():
    assert GetWeather.tool_name() == "get_weather"
    assert SubmitResult.tool_name() == "submit_result"


def test_tool_description_from_docstring():
    assert GetWeather.tool_description() == "Get weather for a location."


def test_tool_parameters_schema():
    schema = GetWeather.parameters_schema()
    assert "properties" in schema
    assert "location" in schema["properties"]


@pytest.mark.asyncio
async def test_tool_execution():
    executor = ToolExecutor([GetWeather])
    tool_call = ToolCall(id="tc_1", name="get_weather", arguments={"location": "SF"})
    result = await executor.execute([tool_call])
    assert len(result.results) == 1
    assert result.results[0].content == "Weather in SF: 72F"
    assert not result.results[0].is_error


@pytest.mark.asyncio
async def test_tool_timeout():
    executor = ToolExecutor([SlowTool])
    tool_call = ToolCall(id="tc_1", name="slow_tool", arguments={})
    result = await executor.execute([tool_call])
    assert len(result.results) == 1
    assert result.results[0].is_error
    assert "timed out" in result.results[0].content


@pytest.mark.asyncio
async def test_tool_requires_confirmation_fails_without_callback():
    """Tools requiring confirmation should fail if no callback provided."""
    executor = ToolExecutor([ConfirmationTool])
    tool_call = ToolCall(id="tc_1", name="confirmation_tool", arguments={"action": "delete"})
    result = await executor.execute([tool_call])
    assert len(result.results) == 1
    assert result.results[0].is_error
    assert "requires confirmation" in result.results[0].content


@pytest.mark.asyncio
async def test_tool_requires_confirmation_with_callback():
    """Tools requiring confirmation should work with callback."""
    executor = ToolExecutor([ConfirmationTool])
    tool_call = ToolCall(id="tc_1", name="confirmation_tool", arguments={"action": "delete"})

    async def confirm(tc: ToolCall) -> bool:
        return True

    result = await executor.execute([tool_call], on_confirmation=confirm)
    assert len(result.results) == 1
    assert not result.results[0].is_error
    assert result.results[0].content == "Executed: delete"


@pytest.mark.asyncio
async def test_tool_requires_confirmation_denied():
    """Tools requiring confirmation should return error if denied."""
    executor = ToolExecutor([ConfirmationTool])
    tool_call = ToolCall(id="tc_1", name="confirmation_tool", arguments={"action": "delete"})

    async def deny(tc: ToolCall) -> bool:
        return False

    result = await executor.execute([tool_call], on_confirmation=deny)
    assert len(result.results) == 1
    assert result.results[0].is_error
    assert "cancelled" in result.results[0].content


def test_executor_get_schemas():
    executor = ToolExecutor([GetWeather])
    schemas = executor.get_schemas_for_provider()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "get_weather"
    assert schemas[0]["description"] == "Get weather for a location."
    assert "input_schema" in schemas[0]
