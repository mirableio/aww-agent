import asyncio
import json
from typing import Any, Awaitable, Callable, Type
from pydantic import ValidationError
from ..core.content import ToolCall, ToolResult
from ..exceptions import ToolExecutionError, ToolNotFoundError, ToolValidationError
from .base import Tool


class ToolExecutor:
    def __init__(self, tools: list[Type[Tool]]):
        self._tools: dict[str, Type[Tool]] = {}
        for tool_cls in tools:
            self._tools[tool_cls.tool_name()] = tool_cls

    def get(self, name: str) -> Type[Tool]:
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found")
        return self._tools[name]

    def list_tools(self) -> list[Type[Tool]]:
        return list(self._tools.values())

    def get_schemas_for_provider(self) -> list[dict[str, Any]]:
        return [
            {
                "name": cls.tool_name(),
                "description": cls.tool_description(),
                "input_schema": cls.parameters_schema(),
            }
            for cls in self._tools.values()
        ]

    async def execute(self, tool_calls: list[ToolCall], *,
                      on_confirmation: Callable[[ToolCall], Awaitable[bool]] | None = None) -> "ToolExecutionResult":
        """Execute tool calls serially. Parallel execution deferred to future version."""
        result = ToolExecutionResult()

        for tc in tool_calls:
            try:
                res = await self._execute_single(tc, on_confirmation)
                result.results.append(res)
            except Exception as e:
                result.errors.append(e)
                result.results.append(ToolResult(tool_use_id=tc.id, content=f"Error: {str(e)}", is_error=True))

        return result

    async def _execute_single(self, tool_call: ToolCall,
                              on_confirmation: Callable[[ToolCall], Awaitable[bool]] | None) -> ToolResult:
        tool_cls = self.get(tool_call.name)

        # Fail closed: requires_confirmation without callback is an error
        if tool_cls.requires_confirmation:
            if on_confirmation is None:
                raise ToolExecutionError(f"Tool '{tool_call.name}' requires confirmation but no callback provided")
            confirmed = await on_confirmation(tool_call)
            if not confirmed:
                return ToolResult(tool_use_id=tool_call.id, content="Tool execution cancelled by user", is_error=True)

        # Instantiate tool with arguments (Pydantic validates)
        try:
            tool_instance = tool_cls.model_validate(tool_call.arguments)
        except ValidationError as e:
            raise ToolValidationError(f"Invalid arguments for tool '{tool_call.name}': {e}")

        # Execute with timeout
        try:
            raw_result = await asyncio.wait_for(tool_instance.run(), timeout=tool_cls.timeout)
        except asyncio.TimeoutError:
            raise ToolExecutionError(f"Tool '{tool_call.name}' timed out after {tool_cls.timeout}s")

        content = json.dumps(raw_result) if isinstance(raw_result, dict) else str(raw_result)
        return ToolResult(tool_use_id=tool_call.id, content=content, is_error=False)


class ToolExecutionResult:
    def __init__(self) -> None:
        self.results: list[ToolResult] = []
        self.errors: list[Exception] = []
