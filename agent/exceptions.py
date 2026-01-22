class AgentError(Exception):
    """Base exception for all framework errors."""
    pass


class ToolNotFoundError(AgentError):
    """Raised when a tool is not found in the registry."""
    pass


class ToolValidationError(AgentError):
    """Raised when tool input validation fails."""
    pass


class ToolExecutionError(AgentError):
    """Raised when tool execution fails."""
    pass


class AdapterError(AgentError):
    """Raised when adapter operations fail."""
    pass
