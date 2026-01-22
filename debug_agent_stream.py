"""Debug script to check agent streaming."""
import asyncio
from agent import Agent, Tool, AnthropicAdapter, TextDelta, ToolCallStart, ToolCallComplete, AgentDone

class GetWeather(Tool):
    """Get weather for a location."""
    location: str

    async def run(self) -> str:
        return f"72°F in {self.location}"

async def main():
    agent = Agent(
        adapter=AnthropicAdapter(),
        tools=[GetWeather],
        system_prompt="Be concise.",
    )

    print("Starting agent stream...")
    async for event in agent.run_stream("Say hello"):
        print(f"Event: {type(event).__name__}")
        if isinstance(event, TextDelta):
            print(f"  text: {repr(event.delta)}")
        elif isinstance(event, AgentDone):
            print(f"  reason: {event.stop_reason}, tokens: {event.total_tokens}")

asyncio.run(main())
