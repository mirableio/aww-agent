"""Centralized default configuration values."""
import os
from dotenv import load_dotenv

# Load .env file (if exists)
load_dotenv()

# Provider-specific API keys (common convention).
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Shared model id used by adapters when provided.
LLM_MODEL = os.getenv("LLM_MODEL")

# LLM runtime defaults
LLM_MAX_TOKENS = 4096
LLM_MAX_RETRIES = 3
LLM_TIMEOUT = 120.0

# Tool defaults
DEFAULT_TOOL_TIMEOUT = 30.0

# Agent loop defaults
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_ERROR_THRESHOLD = 3
