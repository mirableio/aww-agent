from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any
from pydantic import BaseModel, Field
from ..config import DEFAULT_MAX_TOKENS
from ..core.messages import Message


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(BaseModel):
    message: Message
    stop_reason: str
    usage: TokenUsage
    raw_response: Any = Field(default=None, exclude=True)
    model_config = {"arbitrary_types_allowed": True}


class BaseAdapter(ABC):
    @abstractmethod
    async def complete(self, messages: list[Message], *, system: str | None = None,
                       tools: list[Any] | None = None, max_tokens: int = DEFAULT_MAX_TOKENS, **kwargs: Any) -> ModelResponse:
        ...

    @abstractmethod
    def stream(self, messages: list[Message], *, system: str | None = None,
               tools: list[Any] | None = None, max_tokens: int = DEFAULT_MAX_TOKENS, **kwargs: Any) -> AsyncGenerator[Any, None]:
        ...

    @abstractmethod
    def count_tokens(self, messages: list[Message]) -> int:
        """Estimate token count. PLACEHOLDER: uses char/4 heuristic, not accurate for budgeting."""
        ...

    @abstractmethod
    def convert_to_provider(self, messages: list[Message], system: str | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def convert_from_provider(self, response: Any) -> ModelResponse:
        ...

    @abstractmethod
    def convert_tools_to_provider(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tool schemas to provider format. Input is from ToolExecutor.get_schemas_for_provider()."""
        ...
