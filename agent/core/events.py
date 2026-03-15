"""Agent events for streaming."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .content import ToolCall, ToolResult
from .types import StopReason

if TYPE_CHECKING:
    from .messages import Message


@dataclass
class AgentEvent:
    """Base class for agent events."""
    pass


@dataclass
class TextDelta(AgentEvent):
    """Partial text chunk from streaming."""
    delta: str


@dataclass
class ToolCallStart(AgentEvent):
    """Tool call parsed from stream, about to execute."""
    tool_call: ToolCall


@dataclass
class ToolCallComplete(AgentEvent):
    """Tool finished executing."""
    tool_call: ToolCall
    result: ToolResult


@dataclass
class TurnComplete(AgentEvent):
    """Model turn finished (text + any tool calls)."""
    iteration: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class AgentDone(AgentEvent):
    """Agent loop finished."""
    stop_reason: StopReason
    total_tokens: int
    messages: "list[Message]" = field(default_factory=list)
