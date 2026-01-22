from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator

from .types import Role
from .content import (
    ContentBlock, TextContent, ToolCallContent, ToolResultContent,
    ToolCall, ToolResult,
)


class MessageMetadata(BaseModel):
    token_count: int | None = Field(default=None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    model_config = {"frozen": True}


class Message(BaseModel):
    role: Role
    content: list[ContentBlock] = Field(default_factory=list)
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)
    model_config = {"extra": "forbid", "validate_assignment": True}

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v: Any) -> list[ContentBlock]:
        if isinstance(v, str):
            return [TextContent(text=v)]
        return list(v)

    @model_validator(mode="after")
    def validate_role_content_consistency(self) -> "Message":
        if self.role == Role.ASSISTANT:
            for block in self.content:
                if isinstance(block, ToolResultContent):
                    raise ValueError("Assistant messages cannot contain tool_result blocks")
        elif self.role == Role.TOOL:
            for block in self.content:
                if not isinstance(block, ToolResultContent):
                    raise ValueError("Tool messages must only contain tool_result blocks")
        return self

    @classmethod
    def user(cls, text: str, **kwargs: Any) -> "Message":
        return cls(role=Role.USER, content=[TextContent(text=text)], **kwargs)

    @classmethod
    def assistant(cls, text: str | None = None, tool_calls: list[ToolCall] | None = None, **kwargs: Any) -> "Message":
        content: list[ContentBlock] = []
        if text:
            content.append(TextContent(text=text))
        if tool_calls:
            for tc in tool_calls:
                content.append(ToolCallContent(tool_call=tc))
        return cls(role=Role.ASSISTANT, content=content, **kwargs)

    @classmethod
    def tool_result(cls, tool_use_id: str, result: str, is_error: bool = False, **kwargs: Any) -> "Message":
        return cls(
            role=Role.TOOL,
            content=[ToolResultContent(tool_result=ToolResult(tool_use_id=tool_use_id, content=result, is_error=is_error))],
            **kwargs
        )

    @classmethod
    def system(cls, text: str, **kwargs: Any) -> "Message":
        return cls(role=Role.SYSTEM, content=[TextContent(text=text)], **kwargs)

    @property
    def text_content(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextContent))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [block.tool_call for block in self.content if isinstance(block, ToolCallContent)]

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
