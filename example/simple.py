"""Example usage of the agent framework."""
import asyncio
from agent import Agent, Tool

from example.adapter_factory import build_adapter


class GetWeather(Tool):
    """Get current weather for a location."""
    location: str

    async def run(self) -> str:
        return f"Weather in {self.location}: 72°F, sunny"


class Calculate(Tool):
    """Perform a calculation."""
    expression: str

    async def run(self) -> str:
        try:
            result = eval(self.expression)  # noqa: S307
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"


async def main() -> None:
    adapter = build_adapter()
    print(f"Using: {type(adapter).__name__}")
    agent = Agent(
        adapter=adapter,
        tools=[GetWeather, Calculate],
        system_prompt="You are a helpful assistant. Answer questions concisely.",
    )

    result = await agent.run("What's the weather in San Francisco?")
    print(f"Response: {result.text}")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Tokens used: {result.tokens}")


if __name__ == "__main__":
    asyncio.run(main())
