from typing import Any, Literal, Union
from pydantic import BaseModel, Field


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCall(BaseModel):
    id: str = Field(..., description="Unique identifier for this tool call")
    name: str = Field(..., description="Name of the tool to invoke")
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallContent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall


class ToolResult(BaseModel):
    tool_use_id: str = Field(..., description="ID of the tool call this responds to")
    content: str | list[dict[str, Any]] = Field(...)
    is_error: bool = Field(default=False)


class ToolResultContent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_result: ToolResult


ContentBlock = Union[TextContent, ToolCallContent, ToolResultContent]
