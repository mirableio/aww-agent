from collections.abc import AsyncGenerator
from typing import Any, Awaitable, Callable, Type

from ..config import DEFAULT_MAX_TOKENS, DEFAULT_MAX_ITERATIONS, DEFAULT_ERROR_THRESHOLD
from ..core.messages import Message
from ..core.content import ContentBlock, TextContent, ToolCallContent
from ..core.types import Role, StopReason
from ..core.events import (
    AgentEvent, TextDelta, ToolCallStart, ToolCallComplete, TurnComplete, AgentDone
)
from ..adapters.base import BaseAdapter
from ..adapters.anthropic import AnthropicAdapter, StreamResult
from ..tools.base import Tool, SubmitResult
from ..tools.executor import ToolExecutor


class AgentResult:
    def __init__(self, messages: list[Message], stop_reason: StopReason, tokens: int,
                 done_tool_result: str | None = None):
        self.messages = messages
        self.stop_reason = stop_reason
        self.tokens = tokens
        self._done_tool_result = done_tool_result

    @property
    def text(self) -> str:
        """The final output. Prefers done tool result if present, else last assistant text."""
        if self._done_tool_result is not None:
            return self._done_tool_result
        for msg in reversed(self.messages):
            if msg.role.value == "assistant":
                return msg.text_content
        return ""


class Agent:
    def __init__(
        self,
        adapter: BaseAdapter,
        tools: list[Type[Tool]] | None = None,
        *,
        system_prompt: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        error_threshold: int = DEFAULT_ERROR_THRESHOLD,
        include_done_tool: bool = False,
    ):
        self.adapter = adapter
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.error_threshold = error_threshold

        # Build tool list
        all_tools = list(tools) if tools else []
        if include_done_tool:
            all_tools.append(SubmitResult)

        self.executor = ToolExecutor(all_tools)
        self._interrupt_requested = False

    def request_interrupt(self) -> None:
        self._interrupt_requested = True

    async def run(self, prompt: str | list[Message], *,
                  on_message: Callable[[Message], Awaitable[None]] | None = None,
                  on_confirmation: Callable[[Any], Awaitable[bool]] | None = None) -> AgentResult:
        self._interrupt_requested = False

        # Accept string or list of messages
        if isinstance(prompt, str):
            conversation = [Message.user(prompt)]
        else:
            conversation = list(prompt)
        total_tokens = 0
        iteration = 0
        consecutive_errors = 0
        done_tool_name = SubmitResult.tool_name()

        while True:
            iteration += 1

            # Check user interrupt
            if self._interrupt_requested:
                return AgentResult(messages=conversation, stop_reason=StopReason.USER_INTERRUPT, tokens=total_tokens)

            # Check max iterations
            if iteration > self.max_iterations:
                return AgentResult(messages=conversation, stop_reason=StopReason.MAX_ITERATIONS, tokens=total_tokens)

            # Check error threshold
            if consecutive_errors >= self.error_threshold:
                return AgentResult(messages=conversation, stop_reason=StopReason.ERROR_THRESHOLD, tokens=total_tokens)

            # Call model
            tools = self.executor.get_schemas_for_provider() if self.executor.list_tools() else None
            response = await self.adapter.complete(messages=conversation, system=self.system_prompt,
                                                   tools=tools, max_tokens=self.max_tokens)
            total_tokens += response.usage.total_tokens
            conversation.append(response.message)
            if on_message:
                await on_message(response.message)

            # Natural completion: no tool calls
            if not response.message.has_tool_calls:
                return AgentResult(messages=conversation, stop_reason=StopReason.NATURAL_COMPLETION, tokens=total_tokens)

            # Execute tools (including done tool if present)
            tool_calls = response.message.tool_calls
            exec_result = await self.executor.execute(tool_calls, on_confirmation=on_confirmation)

            if exec_result.errors:
                consecutive_errors += len(exec_result.errors)
            else:
                consecutive_errors = 0

            # Check if done tool was called - capture its result
            done_tool_result: str | None = None
            for tc, result in zip(tool_calls, exec_result.results):
                if tc.name == done_tool_name and not result.is_error:
                    done_tool_result = result.content if isinstance(result.content, str) else str(result.content)
                    break

            # Add tool results to conversation
            for result in exec_result.results:
                msg = Message.tool_result(
                    tool_use_id=result.tool_use_id,
                    result=result.content if isinstance(result.content, str) else str(result.content),
                    is_error=result.is_error,
                )
                conversation.append(msg)
                if on_message:
                    await on_message(msg)

            # If done tool was called successfully, stop
            if done_tool_result is not None:
                return AgentResult(messages=conversation, stop_reason=StopReason.DONE_TOOL,
                                   tokens=total_tokens, done_tool_result=done_tool_result)

    async def run_stream(
        self,
        prompt: str | list[Message],
        *,
        on_confirmation: Callable[[Any], Awaitable[bool]] | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run agent loop with streaming, yielding events as they happen."""
        self._interrupt_requested = False

        # Accept string or list of messages
        if isinstance(prompt, str):
            conversation = [Message.user(prompt)]
        else:
            conversation = list(prompt)

        total_tokens = 0
        iteration = 0
        consecutive_errors = 0
        done_tool_name = SubmitResult.tool_name()

        # Check adapter supports streaming
        if not isinstance(self.adapter, AnthropicAdapter):
            raise NotImplementedError("Streaming only supported with AnthropicAdapter")

        while True:
            iteration += 1

            # Check user interrupt
            if self._interrupt_requested:
                yield AgentDone(stop_reason=StopReason.USER_INTERRUPT, total_tokens=total_tokens, messages=conversation)
                return

            # Check max iterations
            if iteration > self.max_iterations:
                yield AgentDone(stop_reason=StopReason.MAX_ITERATIONS, total_tokens=total_tokens, messages=conversation)
                return

            # Check error threshold
            if consecutive_errors >= self.error_threshold:
                yield AgentDone(stop_reason=StopReason.ERROR_THRESHOLD, total_tokens=total_tokens, messages=conversation)
                return

            # Stream from model
            tools = self.executor.get_schemas_for_provider() if self.executor.list_tools() else None
            stream_result: StreamResult | None = None

            async for event in self.adapter.stream_with_events(
                messages=conversation,
                system=self.system_prompt,
                tools=tools,
                max_tokens=self.max_tokens,
            ):
                if isinstance(event, TextDelta):
                    yield event
                elif isinstance(event, StreamResult):
                    stream_result = event

            if stream_result is None:
                yield AgentDone(stop_reason=StopReason.ERROR_THRESHOLD, total_tokens=total_tokens, messages=conversation)
                return

            # Update token count
            total_tokens += stream_result.usage.total_tokens

            # Build assistant message from stream result
            content_blocks: list[ContentBlock] = []
            if stream_result.text:
                content_blocks.append(TextContent(text=stream_result.text))
            for tc in stream_result.tool_calls:
                content_blocks.append(ToolCallContent(tool_call=tc))

            assistant_msg = Message(role=Role.ASSISTANT, content=content_blocks)
            conversation.append(assistant_msg)

            # Yield turn complete
            yield TurnComplete(
                iteration=iteration,
                input_tokens=stream_result.usage.input_tokens,
                output_tokens=stream_result.usage.output_tokens,
            )

            # Natural completion: no tool calls
            if not stream_result.tool_calls:
                yield AgentDone(stop_reason=StopReason.NATURAL_COMPLETION, total_tokens=total_tokens, messages=conversation)
                return

            # Execute tools
            done_tool_result: str | None = None

            for tc in stream_result.tool_calls:
                yield ToolCallStart(tool_call=tc)

                # Execute single tool
                exec_result = await self.executor.execute([tc], on_confirmation=on_confirmation)
                result = exec_result.results[0]

                yield ToolCallComplete(tool_call=tc, result=result)

                if exec_result.errors:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                # Check if done tool
                if tc.name == done_tool_name and not result.is_error:
                    done_tool_result = result.content if isinstance(result.content, str) else str(result.content)

                # Add tool result to conversation
                msg = Message.tool_result(
                    tool_use_id=result.tool_use_id,
                    result=result.content if isinstance(result.content, str) else str(result.content),
                    is_error=result.is_error,
                )
                conversation.append(msg)

            # If done tool was called successfully, stop
            if done_tool_result is not None:
                yield AgentDone(stop_reason=StopReason.DONE_TOOL, total_tokens=total_tokens, messages=conversation)
                return
