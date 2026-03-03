"""Lightweight agent framework for production AI applications."""

from .adapters.anthropic import AnthropicAdapter
from .adapters.openai import OpenAIAdapter
from .core.messages import Message
from .core.events import (
    AgentEvent,
    TextDelta,
    ToolCallStart,
    ToolCallComplete,
    TurnComplete,
    AgentDone,
)
from .tools.base import Tool
from .loop.runner import Agent, AgentResult
from .exceptions import AgentError
from .ui import chat_loop, run_chat

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentResult",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "Message",
    "Tool",
    "AgentError",
    # Events
    "AgentEvent",
    "TextDelta",
    "ToolCallStart",
    "ToolCallComplete",
    "TurnComplete",
    "AgentDone",
    # UI
    "chat_loop",
    "run_chat",
]
