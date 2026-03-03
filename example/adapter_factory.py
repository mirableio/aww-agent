"""Shared adapter selection for examples."""

import os

from agent import AnthropicAdapter, OpenAIAdapter


def build_adapter() -> AnthropicAdapter | OpenAIAdapter:
    provider = os.getenv("AGENT_PROVIDER", "anthropic").strip().lower()
    if provider == "openai":
        return OpenAIAdapter()
    if provider == "anthropic":
        return AnthropicAdapter()
    raise ValueError("AGENT_PROVIDER must be 'anthropic' or 'openai'")
