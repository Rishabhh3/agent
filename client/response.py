from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

@dataclass
class TextDelta:
    content: str

    def __str__(self):
        return self.content

@dataclass
class EventType(str, Enum):
    TEXT_DELTA = "text_delta"
    MESSAGE_COMPLETE  ="message_complete"
    ERROR = "error"

@dataclass
class TokenUsage:   # this class helps us determine how much time we have to wait before we can do any sort of context compaction
    prompt_tokens :int = 0
    completion_tokens : int = 0
    total_tokens : int = 0
    cached_tokens : int = 0

    def __add__(self, other : TokenUsage): # annotation allows us to use token usage inside token usage
        return TokenUsage(
            prompt_tokens= self.prompt_tokens + other.prompt_tokens,
            completion_tokens= self.completion_tokens + other.completion_tokens,
            total_tokens= self.total_tokens + other.total_tokens,
            cached_tokens= self.cached_tokens + other.cached_tokens,

        )

@dataclass
class StreamEvent:
    type : EventType
    text_delta : TextDelta | None = None # it can be none, because if no text is given by llm, it has no text delta
    error : str | None = None # default to null
    finish_reason : str | None = None
    usage : TokenUsage | None = None    # it might happen the llm dont send the usage, depend on provider
    
