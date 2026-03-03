# aww-agent

Lightweight Python agent framework. Minimal, transparent, async-native.

## Install

```bash
uv sync
```

## Quick Start

```python
import asyncio
from agent import Agent, Tool, AnthropicAdapter, OpenAIAdapter

class GetWeather(Tool):
    """Get weather for a location."""
    location: str

    async def run(self) -> str:
        return f"72°F in {self.location}"

agent = Agent(
    adapter=AnthropicAdapter(),  # replace with OpenAIAdapter() to use OpenAI Responses API
    tools=[GetWeather],
    system_prompt="Be concise.",
)

async def main():
    result = await agent.run("Weather in SF?")
    print(result.text)

asyncio.run(main())
```

## Streaming

```python
from agent import Agent, AnthropicAdapter, OpenAIAdapter, TextDelta, ToolCallStart, ToolCallComplete, AgentDone

agent = Agent(adapter=AnthropicAdapter())  # or OpenAIAdapter()

async for event in agent.run_stream("Hello"):
    match event:
        case TextDelta(delta=text):
            print(text, end="", flush=True)
        case ToolCallStart(tool_call=tc):
            print(f"\n[{tc.name}...]")
        case ToolCallComplete(result=res):
            print(f"[Result: {res.content}]")
        case AgentDone(stop_reason=reason):
            print(f"\nDone: {reason.value}")
```

## Tools

Tools are Pydantic models with a `run()` method:

```python
class SendEmail(Tool):
    """Send an email."""
    to: str
    subject: str
    body: str

    requires_confirmation = True  # asks user before executing
    timeout = 60.0  # seconds

    async def run(self) -> str:
        # send email...
        return f"Sent to {self.to}"
```

## Configuration

Create `.env`:
```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
LLM_MODEL=...
```

Or pass directly:
```python
AnthropicAdapter(api_key="sk-ant-...", model="claude-sonnet-4-5")
# Replace with OpenAIAdapter(api_key="sk-proj-...", model="gpt-5-mini") for OpenAI.
```

## Agent Options

```python
Agent(
    adapter=AnthropicAdapter(),
    tools=[MyTool],
    system_prompt="You are helpful.",
    max_tokens=4096,        # per response
    max_iterations=25,      # loop limit
    error_threshold=3,      # consecutive errors before stop
    include_done_tool=True, # adds submit_result tool
)
```

## Stop Reasons

- `NATURAL_COMPLETION` - model finished without tool calls
- `DONE_TOOL` - model called `submit_result`
- `MAX_ITERATIONS` - hit iteration limit
- `ERROR_THRESHOLD` - too many consecutive tool errors
- `USER_INTERRUPT` - `agent.request_interrupt()` called

## Console UI

Built-in interactive chat with Rich:

```python
from agent import Tool, run_chat

class MyTool(Tool):
    """My custom tool."""
    param: str
    async def run(self) -> str:
        return f"Result: {self.param}"

run_chat(tools=[MyTool], system_prompt="Be helpful.")
```

For more control:
```python
from agent import Agent, AnthropicAdapter, chat_loop
import asyncio

agent = Agent(adapter=AnthropicAdapter(), tools=[MyTool])
asyncio.run(chat_loop(agent))
```

Run the example: `uv run example/console.py`

Controls: Type + Enter, ESC to stop generation, Ctrl+D to quit.

## API

| Class | Description |
|-------|-------------|
| `Agent` | Main agent loop |
| `Tool` | Base class for tools |
| `AnthropicAdapter` | Claude API adapter |
| `OpenAIAdapter` | OpenAI Responses API adapter |
| `Message` | Conversation message |
| `AgentResult` | Result from `run()` |
| `run_chat()` | One-liner interactive console |
| `chat_loop()` | Async chat loop for custom agents |

| Event | Description |
|-------|-------------|
| `TextDelta` | Streaming text chunk |
| `ToolCallStart` | Tool execution starting |
| `ToolCallComplete` | Tool finished |
| `TurnComplete` | Model turn done |
| `AgentDone` | Agent loop finished |

## License

MIT
