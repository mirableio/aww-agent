import re
from typing import Any, ClassVar
from pydantic import BaseModel
from ..config import DEFAULT_TOOL_TIMEOUT


class Tool(BaseModel):
    """Base class for all tools. Subclass and implement run()."""

    # Class-level options (override in subclass)
    timeout: ClassVar[float] = DEFAULT_TOOL_TIMEOUT
    requires_confirmation: ClassVar[bool] = False

    @classmethod
    def tool_name(cls) -> str:
        """Convert class name to snake_case for tool name."""
        name = cls.__name__
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

    @classmethod
    def tool_description(cls) -> str:
        """Get description from docstring."""
        return cls.__doc__ or ""

    @classmethod
    def parameters_schema(cls) -> dict[str, Any]:
        """Generate JSON schema for parameters (excludes ClassVars)."""
        return cls.model_json_schema()

    async def run(self) -> str | dict[str, Any]:
        """Execute the tool. Override in subclass."""
        raise NotImplementedError


class SubmitResult(Tool):
    """Call this tool when you have completed the task and want to submit your final answer."""

    result: str

    async def run(self) -> str:
        return self.result
