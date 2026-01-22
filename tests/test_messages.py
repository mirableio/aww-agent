"""Tests for message abstraction."""
import pytest
from agent.core.messages import Message
from agent.core.content import TextContent, ToolCallContent, ToolCall
from agent.core.types import Role


def test_message_user_creation():
    msg = Message.user("Hello")
    assert msg.role == Role.USER
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextContent)
    assert msg.content[0].text == "Hello"


def test_message_assistant_with_text():
    msg = Message.assistant(text="Hi there")
    assert msg.role == Role.ASSISTANT
    assert msg.text_content == "Hi there"
    assert not msg.has_tool_calls


def test_message_assistant_with_tool_calls():
    tool_call = ToolCall(id="tc_1", name="get_weather", arguments={"location": "SF"})
    msg = Message.assistant(text="Let me check", tool_calls=[tool_call])
    assert msg.role == Role.ASSISTANT
    assert msg.text_content == "Let me check"
    assert msg.has_tool_calls
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "get_weather"


def test_message_content_normalization():
    """String content should be normalized to TextContent."""
    msg = Message(role=Role.USER, content="Hello")  # type: ignore[arg-type]
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextContent)
    assert msg.content[0].text == "Hello"


def test_message_tool_result():
    msg = Message.tool_result(tool_use_id="tc_1", result="72F sunny")
    assert msg.role == Role.TOOL
    assert len(msg.content) == 1


def test_assistant_cannot_have_tool_results():
    """Assistant messages should not contain tool_result blocks."""
    from agent.core.content import ToolResultContent, ToolResult

    with pytest.raises(ValueError, match="cannot contain tool_result"):
        Message(role=Role.ASSISTANT, content=[
            ToolResultContent(tool_result=ToolResult(tool_use_id="tc_1", content="result"))
        ])


def test_tool_messages_only_tool_results():
    """Tool messages should only contain tool_result blocks."""
    with pytest.raises(ValueError, match="must only contain tool_result"):
        Message(role=Role.TOOL, content=[TextContent(text="hello")])
