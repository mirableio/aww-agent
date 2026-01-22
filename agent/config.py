"""Centralized default configuration values."""
import os
from dotenv import load_dotenv

# Load .env file (if exists)
load_dotenv()

# API keys from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model defaults
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 4096

# Adapter defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 120.0

# Tool defaults
DEFAULT_TOOL_TIMEOUT = 30.0

# Agent loop defaults
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_ERROR_THRESHOLD = 3
