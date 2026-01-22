"""Debug script to check Anthropic streaming events."""
import asyncio
from anthropic import AsyncAnthropic
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print("Starting stream...")
    async with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello in 5 words"}],
    ) as stream:
        async for event in stream:
            event_type = getattr(event, "type", type(event).__name__)
            print(f"Event: {event_type}")
            if hasattr(event, "delta"):
                print(f"  delta: {event.delta}")
            if hasattr(event, "text"):
                print(f"  text: {event.text}")

asyncio.run(main())
