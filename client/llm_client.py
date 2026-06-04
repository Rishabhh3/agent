from openai import AsyncOpenAI
from typing import Any, AsyncGenerator
from client.response import *

class LLMClient:
    def __init__(self) -> None:
        self._client : AsyncOpenAI | None = None

    def get_client(self)-> AsyncOpenAI:
        if self._client == None:
            # either client was never instantized or its dependency was closed
            self._client = AsyncOpenAI(
                api_key="",
                base_url= "https://openrouter.ai/api/v1",
            )
        return self._client     # only create one instance, if client is not null just return the client

    async def close(self) ->None:
        if self._client:    # just checks if client is present then close the connection
            await self._client.close()
            self._client = None

    async def chat_completion(self, 
                messages: list[dict[str, Any]], 
                stream: bool = True
                )-> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()
        kwargs = {
            "model" : "poolside/laguna-xs.2:free",
            "messages" : messages,
            "stream" : stream,
        }
        if stream:
            self._stream_response() # these are protected ones that are only present inside this
        else:
            event = await self._non_stream_response(client, kwargs)
            yield event
        return

    async def _stream_response(self):
        pass

    async def _non_stream_response(self, 
                                   client: AsyncOpenAI, 
                                   kwargs: dict[str,Any]) -> StreamEvent:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]    # only intrested in first message
        message = choice.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(content = message.content)

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens= response.usage.prompt_tokens,
                completion_tokens= response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )

        return StreamEvent(
            type=EventType.MESSAGE_COMPLETE,
            text_delta= text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )

