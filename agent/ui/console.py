"""Reusable console chat UI with Rich."""

import asyncio
import sys
import termios
import tty
import readline  # noqa: F401 - enables line editing for input()

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text

from ..adapters.anthropic import AnthropicAdapter
from ..core.events import AgentDone, TextDelta, ToolCallComplete, ToolCallStart
from ..core.messages import Message
from ..loop.runner import Agent
from ..tools.base import Tool

console = Console()


class KeyMonitor:
    """Monitor for Esc key during streaming."""

    def __init__(self) -> None:
        self.triggered = False
        self._old_settings: list | None = None

    def start(self) -> None:
        """Enter raw mode to capture keypresses."""
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.triggered = False

    def stop(self) -> None:
        """Restore terminal settings."""
        if self._old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def check(self) -> bool:
        """Check if Esc was pressed (non-blocking)."""
        import select

        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "\x1b":  # Esc
                self.triggered = True
        return self.triggered


async def chat_loop(agent: Agent) -> None:
    """Run the main chat loop."""
    console.print("[bold]Chat started.[/bold] [dim]Esc to stop, Ctrl+D to exit[/dim]\n")

    key_monitor = KeyMonitor()
    conversation: list[Message] = []

    while True:
        try:
            user_input = input("> ")
        except EOFError:
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        conversation.append(Message.user(user_input))
        text_buffer = ""
        total_tokens = 0

        key_monitor.start()
        try:
            with Live(Spinner("dots", text="[cyan]Thinking...[/cyan]"), console=console, transient=True) as live:
                async for event in agent.run_stream(conversation):
                    if key_monitor.check():
                        agent.request_interrupt()
                        break

                    if isinstance(event, TextDelta):
                        text_buffer += event.delta
                        live.update(Padding(Text(text_buffer), (0, 1), style="on grey23", expand=True))

                    elif isinstance(event, ToolCallStart):
                        # Flush text before tool call
                        if text_buffer.strip():
                            live.console.print(Padding(Markdown(text_buffer), (0, 1), style="on grey23", expand=True))
                            text_buffer = ""
                        live.update(Spinner("dots", text=f"[cyan]● {event.tool_call.name}...[/cyan]"))

                    elif isinstance(event, ToolCallComplete):
                        args = ", ".join(f"{k}={v!r}" for k, v in event.tool_call.arguments.items())
                        result_text = (
                            event.result.content if isinstance(event.result.content, str) else str(event.result.content)
                        )
                        if len(result_text) > 150:
                            result_text = result_text[:150] + "..."
                        live.console.print(Text(f"● {event.tool_call.name}({args}) → {result_text}", style="yellow"))
                        live.update(Spinner("dots", text="[cyan]Thinking...[/cyan]"))

                    elif isinstance(event, AgentDone):
                        total_tokens = event.total_tokens
                        conversation = event.messages
        finally:
            key_monitor.stop()

        if text_buffer:
            console.print(Padding(Markdown(text_buffer), (0, 1), style="on grey23", expand=True))

        status = "(stopped) " if key_monitor.triggered else ""
        console.print(f"[dim]{status}{total_tokens} tokens[/dim]", justify="right")


def run_chat(
    tools: list[type[Tool]] | None = None,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
) -> None:
    """Run interactive chat with optional tools.

    Args:
        tools: List of Tool classes to make available to the agent.
        system_prompt: System prompt for the agent.
        model: Model name for the adapter. If None, uses adapter default.
    """
    adapter = AnthropicAdapter(model=model) if model else AnthropicAdapter()
    agent = Agent(
        adapter=adapter,
        tools=tools or [],
        system_prompt=system_prompt,
    )
    try:
        asyncio.run(chat_loop(agent))
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
