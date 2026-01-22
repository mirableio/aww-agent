import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from anthropic import AsyncAnthropic

from ..config import (
    ANTHROPIC_API_KEY, DEFAULT_MODEL, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, DEFAULT_MAX_TOKENS
)
from ..core.messages import Message
from ..core.content import TextContent, ToolCallContent, ToolResultContent, ToolCall
from ..core.types import Role
from ..core.events import TextDelta
from .base import BaseAdapter, ModelResponse, TokenUsage


@dataclass
class StreamedToolCall:
    """A tool call accumulated from streaming."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class StreamResult:
    """Final result from streaming a turn."""
    text: str
    tool_calls: list[ToolCall]
    usage: TokenUsage
    stop_reason: str


class AnthropicAdapter(BaseAdapter):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = ANTHROPIC_API_KEY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._client: AsyncAnthropic | None = None

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(
                api_key=self.api_key,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
        return self._client

    async def complete(self, messages: list[Message], *, system: str | None = None,
                       tools: list[Any] | None = None, max_tokens: int = DEFAULT_MAX_TOKENS, **kwargs: Any) -> ModelResponse:
        request_params: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "max_tokens": max_tokens,
            "messages": self._convert_messages_to_anthropic(messages),
        }
        if system:
            request_params["system"] = system
        if tools:
            request_params["tools"] = self.convert_tools_to_provider(tools)
        request_params.update(kwargs)
        response = await self.client.messages.create(**request_params)
        return self.convert_from_provider(response)

    async def stream(self, messages: list[Message], *, system: str | None = None,
                     tools: list[Any] | None = None, max_tokens: int = DEFAULT_MAX_TOKENS, **kwargs: Any) -> AsyncGenerator[Any, None]:
        request_params: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "max_tokens": max_tokens,
            "messages": self._convert_messages_to_anthropic(messages),
        }
        if system:
            request_params["system"] = system
        if tools:
            request_params["tools"] = self.convert_tools_to_provider(tools)
        request_params.update(kwargs)
        async with self.client.messages.stream(**request_params) as stream:
            async for chunk in stream:
                yield chunk

    async def stream_with_events(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> AsyncGenerator[TextDelta | StreamResult, None]:
        """Stream and yield parsed events. Yields TextDelta for each text chunk, then StreamResult at end."""
        request_params: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "max_tokens": max_tokens,
            "messages": self._convert_messages_to_anthropic(messages),
        }
        if system:
            request_params["system"] = system
        if tools:
            request_params["tools"] = self.convert_tools_to_provider(tools)
        request_params.update(kwargs)

        text_content = ""
        tool_calls: list[ToolCall] = []
        current_tool_id: str | None = None
        current_tool_name: str | None = None
        current_tool_json: str = ""
        usage: TokenUsage | None = None
        stop_reason: str = "end_turn"

        async with self.client.messages.stream(**request_params) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        current_tool_id = getattr(block, "id", None)
                        current_tool_name = getattr(block, "name", None)
                        current_tool_json = ""

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None) if delta else None
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "")
                        text_content += text
                        yield TextDelta(delta=text)
                    elif delta_type == "input_json_delta":
                        current_tool_json += getattr(delta, "partial_json", "")

                elif event_type == "content_block_stop":
                    if current_tool_id and current_tool_name:
                        try:
                            args = json.loads(current_tool_json) if current_tool_json else {}
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append(ToolCall(
                            id=current_tool_id,
                            name=current_tool_name,
                            arguments=args,
                        ))
                        current_tool_id = None
                        current_tool_name = None
                        current_tool_json = ""

                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    stop_reason = getattr(delta, "stop_reason", "end_turn") or "end_turn"
                    event_usage = getattr(event, "usage", None)
                    if event_usage:
                        usage = TokenUsage(
                            input_tokens=getattr(event_usage, "input_tokens", 0),
                            output_tokens=getattr(event_usage, "output_tokens", 0),
                        )

                elif event_type == "message_start":
                    message = getattr(event, "message", None)
                    msg_usage = getattr(message, "usage", None) if message else None
                    if msg_usage:
                        usage = TokenUsage(
                            input_tokens=getattr(msg_usage, "input_tokens", 0),
                            output_tokens=getattr(msg_usage, "output_tokens", 0),
                        )

        # Get final usage from stream if available
        final_usage = usage or TokenUsage(input_tokens=0, output_tokens=0)

        yield StreamResult(
            text=text_content,
            tool_calls=tool_calls,
            usage=final_usage,
            stop_reason=stop_reason,
        )

    def count_tokens(self, messages: list[Message]) -> int:
        """Rough estimate only. TODO: Use Anthropic's token counting API for accuracy."""
        total_chars = sum(len(msg.text_content) for msg in messages)
        return total_chars // 4

    def convert_to_provider(self, messages: list[Message], system: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"messages": self._convert_messages_to_anthropic(messages)}
        if system:
            result["system"] = system
        return result

    def convert_from_provider(self, response: Any) -> ModelResponse:
        content_blocks: list[Any] = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextContent(text=block.text))
            elif block.type == "tool_use":
                content_blocks.append(ToolCallContent(
                    tool_call=ToolCall(id=block.id, name=block.name, arguments=block.input)
                ))
        message = Message(role=Role.ASSISTANT, content=content_blocks)
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        )
        return ModelResponse(message=message, stop_reason=response.stop_reason, usage=usage, raw_response=response)

    def convert_tools_to_provider(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Tools already in Anthropic format from ToolExecutor.get_schemas_for_provider()
        return tools

    def _convert_messages_to_anthropic(self, messages: list[Message]) -> list[dict[str, Any]]:
        anthropic_messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue
            if msg.role == Role.TOOL:
                for block in msg.content:
                    if isinstance(block, ToolResultContent):
                        result_block: dict[str, Any] = {
                            "type": "tool_result",
                            "tool_use_id": block.tool_result.tool_use_id,
                            "content": block.tool_result.content,
                        }
                        if block.tool_result.is_error:
                            result_block["is_error"] = True
                        pending_tool_results.append(result_block)
                continue

            if pending_tool_results:
                anthropic_messages.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

            if msg.role == Role.USER:
                anthropic_messages.append({"role": "user", "content": self._convert_content_to_anthropic(msg.content)})
            elif msg.role == Role.ASSISTANT:
                anthropic_messages.append({"role": "assistant", "content": self._convert_content_to_anthropic(msg.content)})

        if pending_tool_results:
            anthropic_messages.append({"role": "user", "content": pending_tool_results})
        return anthropic_messages

    def _convert_content_to_anthropic(self, content: list[Any]) -> list[dict[str, Any]] | str:
        if len(content) == 1 and isinstance(content[0], TextContent):
            return content[0].text
        result: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, TextContent):
                result.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolCallContent):
                result.append({"type": "tool_use", "id": block.tool_call.id, "name": block.tool_call.name, "input": block.tool_call.arguments})
        return result
