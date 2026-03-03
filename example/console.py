"""Example console chat with demo tools."""

import os
from agent import Tool, run_chat


class GetWeather(Tool):
    """Get current weather for a location."""

    location: str

    async def run(self) -> str:
        return f"72°F and sunny in {self.location}"


class Calculate(Tool):
    """Evaluate a math expression."""

    expression: str

    async def run(self) -> str:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
        result = eval(self.expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)


if __name__ == "__main__":
    provider = os.getenv("AGENT_PROVIDER", "anthropic").strip().lower()
    print(f"Using provider: {provider}")
    run_chat(
        tools=[GetWeather, Calculate],
        system_prompt="Be helpful and concise.",
        provider=provider,
    )
