from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(str, Enum):
    NATURAL_COMPLETION = "natural_completion"
    DONE_TOOL = "done_tool"
    MAX_ITERATIONS = "max_iterations"
    ERROR_THRESHOLD = "error_threshold"
    USER_INTERRUPT = "user_interrupt"
    # TOKEN_BUDGET = "token_budget"  # Deferred: requires accurate token counting
