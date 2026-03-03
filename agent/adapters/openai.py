import json
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from ..config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
    LLM_MAX_TOKENS,
)
from ..core.content import TextContent, ToolCall, ToolCallContent, ToolResultContent
from ..core.events import TextDelta
from ..core.messages import Message
from ..core.types import Role
from ..exceptions import AdapterError
from .base import BaseAdapter, ModelResponse, StreamResult, TokenUsage


class OpenAIAdapter(BaseAdapter):
    def __init__(
        self,
        model: str | None = LLM_MODEL,
        api_key: str | None = OPENAI_API_KEY,
        max_retries: int = LLM_MAX_RETRIES,
        timeout: float = LLM_TIMEOUT,
    ):
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
        return self._client

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[Any] | None = None,
        max_tokens: int = LLM_MAX_TOKENS,
        **kwargs: Any,
    ) -> ModelResponse:
        model = kwargs.pop("model", self.model)
        if not model:
            raise AdapterError("OpenAI model not configured. Set LLM_MODEL or pass model=...")

        request_params: dict[str, Any] = {
            "model": model,
            "max_output_tokens": max_tokens,
            "input": self._convert_messages_to_responses_input(messages),
        }
        if system:
            request_params["instructions"] = system
        if tools:
            request_params["tools"] = self.convert_tools_to_provider(tools)
        request_params.update(kwargs)

        response = await self.client.responses.create(**request_params)
        return self.convert_from_provider(response)

    async def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[Any] | None = None,
        max_tokens: int = LLM_MAX_TOKENS,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        model = kwargs.pop("model", self.model)
        if not model:
            raise AdapterError("OpenAI model not configured. Set LLM_MODEL or pass model=...")

        request_params: dict[str, Any] = {
            "model": model,
            "max_output_tokens": max_tokens,
            "input": self._convert_messages_to_responses_input(messages),
            "stream": True,
        }
        if system:
            request_params["instructions"] = system
        if tools:
            request_params["tools"] = self.convert_tools_to_provider(tools)
        request_params.update(kwargs)

        stream = await self.client.responses.create(**request_params)
        async for event in stream:
            yield event

    async def stream_with_events(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[Any] | None = None,
        max_tokens: int = LLM_MAX_TOKENS,
        **kwargs: Any,
    ) -> AsyncGenerator[TextDelta | StreamResult, None]:
        text_content = ""
        final_response: Any | None = None

        async for event in self.stream(
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            **kwargs,
        ):
            event_type = getattr(event, "type", None)

            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    text_content += delta
                    yield TextDelta(delta=delta)
            elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
                event_response = getattr(event, "response", None)
                if event_response is not None:
                    final_response = event_response

        if final_response is None:
            raise AdapterError("OpenAI stream finished without final response event")

        converted = self.convert_from_provider(final_response)
        final_text = text_content if text_content else converted.message.text_content

        yield StreamResult(
            text=final_text,
            tool_calls=converted.message.tool_calls,
            usage=converted.usage,
            stop_reason=converted.stop_reason,
        )

    def count_tokens(self, messages: list[Message]) -> int:
        """Rough estimate only. Uses char/4 heuristic."""
        total_chars = sum(len(msg.text_content) for msg in messages)
        return total_chars // 4

    def convert_to_provider(self, messages: list[Message], system: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"input": self._convert_messages_to_responses_input(messages)}
        if system:
            result["instructions"] = system
        return result

    def convert_from_provider(self, response: Any) -> ModelResponse:
        content_blocks: list[Any] = []

        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                for part in getattr(item, "content", []) or []:
                    part_type = getattr(part, "type", None)
                    if part_type in {"output_text", "text", "input_text"}:
                        text = getattr(part, "text", None)
                        if text:
                            content_blocks.append(TextContent(text=text))

            elif item_type == "function_call":
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or ""
                name = getattr(item, "name", None) or ""
                arguments = self._parse_arguments(getattr(item, "arguments", None))

                if not call_id:
                    call_id = f"call_{len(content_blocks)}"

                content_blocks.append(
                    ToolCallContent(tool_call=ToolCall(id=call_id, name=name, arguments=arguments))
                )

            elif item_type == "output_text":
                text = getattr(item, "text", None)
                if text:
                    content_blocks.append(TextContent(text=text))

        message = Message(role=Role.ASSISTANT, content=content_blocks)
        usage = self._convert_usage(getattr(response, "usage", None))
        stop_reason = self._extract_stop_reason(response)
        return ModelResponse(message=message, stop_reason=stop_reason, usage=usage, raw_response=response)

    def convert_tools_to_provider(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            converted.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                    "strict": False,
                }
            )
        return converted

    def _convert_messages_to_responses_input(self, messages: list[Message]) -> list[dict[str, Any]]:
        response_items: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue

            if msg.role == Role.USER:
                text = msg.text_content
                if text:
                    response_items.append(self._message_item(role="user", text=text))
                continue

            if msg.role == Role.ASSISTANT:
                text = msg.text_content
                if text:
                    response_items.append(self._message_item(role="assistant", text=text))

                for block in msg.content:
                    if isinstance(block, ToolCallContent):
                        response_items.append(
                            {
                                "type": "function_call",
                                "call_id": block.tool_call.id,
                                "name": block.tool_call.name,
                                "arguments": json.dumps(block.tool_call.arguments),
                            }
                        )
                continue

            if msg.role == Role.TOOL:
                for block in msg.content:
                    if isinstance(block, ToolResultContent):
                        response_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": block.tool_result.tool_use_id,
                                "output": self._tool_result_to_output(block.tool_result.content),
                            }
                        )

        return response_items

    @staticmethod
    def _message_item(role: str, text: str) -> dict[str, Any]:
        return {
            "role": role,
            "content": text,
        }

    @staticmethod
    def _tool_result_to_output(content: str | list[dict[str, Any]]) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content)

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _extract_stop_reason(response: Any) -> str:
        status = getattr(response, "status", None)
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) if details else None
            if reason:
                return f"incomplete:{reason}"
        if status:
            return str(status)
        return "unknown"

    @staticmethod
    def _convert_usage(usage: Any) -> TokenUsage:
        if not usage:
            return TokenUsage(input_tokens=0, output_tokens=0)

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        cache_read_tokens = 0
        input_details = getattr(usage, "input_tokens_details", None)
        if input_details:
            cache_read_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
        )
