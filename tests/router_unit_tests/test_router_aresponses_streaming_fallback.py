"""
Unit tests for the Responses-API streaming-fallback helpers added to Router
in PR #28215 (fix(router): wrap aresponses streaming iterator for mid-stream
fallbacks).

Targets the four helpers introduced on Router:
  - _extract_partial_responses_usage
  - _combine_responses_fallback_usage
  - _build_responses_continuation_input
  - _aresponses_streaming_iterator
"""

from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from litellm import Router
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


def _make_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                },
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-test",
                },
            },
        ]
    )


def _make_completed_event(
    input_tokens: int, output_tokens: int, total_tokens: int
) -> ResponseCompletedEvent:
    response = ResponsesAPIResponse.model_construct(
        usage=ResponseAPIUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    )
    return ResponseCompletedEvent.model_construct(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=response,
    )


# -------- _extract_partial_responses_usage --------


def test_extract_partial_responses_usage_native_completed():
    """Native path: completed_response carries usage → returned as-is."""
    completed = _make_completed_event(11, 7, 18)
    source = MagicMock()
    source.completed_response = completed

    usage = Router._extract_partial_responses_usage(source)
    assert usage is not None
    assert usage.input_tokens == 11
    assert usage.output_tokens == 7
    assert usage.total_tokens == 18


def test_extract_partial_responses_usage_no_completed_response():
    """Native path: no completed_response → returns None."""
    source = MagicMock()
    source.completed_response = None

    usage = Router._extract_partial_responses_usage(source)
    assert usage is None


def test_extract_partial_responses_usage_bridge_iterator_no_completed_response():
    """
    Regression for #35411: the bridge iterator
    (LiteLLMCompletionStreamingIterator) overrides __init__ without calling
    super().__init__(), so completed_response was never set until the stream
    reached RESPONSE_COMPLETED. On a mid-stream provider error (before
    completion) the fallback recovery path read source_iterator.completed_response
    and raised AttributeError, masking the real provider error and bypassing
    fallbacks. The attribute must always exist and default to None.
    """
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )

    wrapper = MagicMock()
    wrapper.logging_obj = MagicMock()
    iterator = LiteLLMCompletionStreamingIterator(
        model="anthropic/claude-sonnet-4-5",
        litellm_custom_stream_wrapper=wrapper,
        request_input="hi",
        responses_api_request={},
    )

    assert iterator.completed_response is None
    # No chat chunks collected yet and no completed_response → must return
    # None instead of raising AttributeError.
    assert Router._extract_partial_responses_usage(iterator) is None


# -------- _combine_responses_fallback_usage --------


def test_combine_responses_fallback_usage_sums_completed_event():
    """Partial-stream usage is summed into the fallback event's usage."""
    fallback_event = _make_completed_event(5, 3, 8)
    partial = ResponseAPIUsage(input_tokens=11, output_tokens=7, total_tokens=18)

    Router._combine_responses_fallback_usage(fallback_event, partial)

    combined = fallback_event.response.usage
    assert combined is not None
    assert combined.input_tokens == 16
    assert combined.output_tokens == 10
    assert combined.total_tokens == 26


def test_combine_responses_fallback_usage_passthrough_for_unknown_event():
    """Events that are not completed/failed/incomplete are not mutated."""
    other = MagicMock()  # not a ResponseCompletedEvent etc. → isinstance false
    partial = ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    Router._combine_responses_fallback_usage(other, partial)
    # No mutation expected on the unknown event — call is a no-op.


# -------- _build_responses_continuation_input --------


def test_build_responses_continuation_input_from_string():
    out = Router._build_responses_continuation_input(
        "Hello world", "partial assistant text"
    )
    assert len(out) == 3
    assert out[0]["role"] == "user"
    assert out[0]["content"][0]["text"] == "Hello world"
    assert out[1]["role"] == "developer"
    assert out[2]["role"] == "assistant"
    assert out[2]["content"][0]["text"] == "partial assistant text"


def test_build_responses_continuation_input_from_list_preserves_items():
    existing: List[Any] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "msg1"}],
        }
    ]
    out = Router._build_responses_continuation_input(existing, "partial")
    assert len(out) == 3
    assert out[0]["content"][0]["text"] == "msg1"
    assert out[1]["role"] == "developer"
    assert out[2]["role"] == "assistant"


def test_build_responses_continuation_input_from_none():
    out = Router._build_responses_continuation_input(None, "partial")
    assert len(out) == 2
    assert out[0]["role"] == "developer"
    assert out[1]["role"] == "assistant"


# -------- _aresponses_streaming_iterator (passthrough smoke test) --------


@pytest.mark.asyncio
async def test_aresponses_streaming_iterator_passthrough():
    """
    Without MidStreamFallbackError, the wrapper yields source events
    unchanged and returns a BaseResponsesAPIStreamingIterator subclass.
    """
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    events = [_make_completed_event(1, 1, 2)]

    class _FakeSource:
        """Minimal source iterator. Provides every attribute the wrapper
        constructor reads from source_iterator."""

        def __init__(self) -> None:
            self._i = 0
            self.completed_response = None
            self.response = MagicMock()
            self.model = "openai/gpt-4o-mini"
            self.logging_obj = MagicMock()
            self.responses_api_provider_config = MagicMock()
            self.start_time = 0.0
            self.litellm_metadata = {}
            self.custom_llm_provider = "openai"
            self.request_data = {}
            self.call_type = "aresponses"
            self._hidden_params: dict = {}

        def __aiter__(self) -> AsyncIterator[Any]:
            return self

        async def __anext__(self):
            if self._i >= len(events):
                raise StopAsyncIteration
            ev = events[self._i]
            self._i += 1
            return ev

        async def aclose(self):
            return None

    router = _make_router()
    source = _FakeSource()

    wrapper = await router._aresponses_streaming_iterator(
        source, initial_kwargs={"model": "primary"}
    )
    assert isinstance(wrapper, BaseResponsesAPIStreamingIterator)

    collected = [ev async for ev in wrapper]
    assert len(collected) == 1
    assert collected[0].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED


# -------- _aresponses_with_streaming_fallbacks --------


@pytest.mark.asyncio
async def test_aresponses_with_streaming_fallbacks_non_streaming_passthrough():
    """Non-streaming response is returned unchanged, no wrap."""
    router = _make_router()
    plain_response = MagicMock()

    async def fake_original(**_kwargs):
        return plain_response

    with patch.object(
        router,
        "_ageneric_api_call_with_fallbacks_helper",
        new=AsyncMock(return_value=plain_response),
    ):
        out = await router._aresponses_with_streaming_fallbacks(
            original_function=fake_original,
            model="primary",
            stream=False,
        )
    assert out is plain_response


@pytest.mark.asyncio
async def test_aresponses_with_streaming_fallbacks_wraps_streaming_iterator():
    """Streaming response is wrapped via _aresponses_streaming_iterator."""
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    router = _make_router()
    streaming_iter = MagicMock(spec=BaseResponsesAPIStreamingIterator)
    wrapped = MagicMock(spec=BaseResponsesAPIStreamingIterator)

    async def fake_original(**_kwargs):
        return streaming_iter

    with patch.object(
        router,
        "_ageneric_api_call_with_fallbacks_helper",
        new=AsyncMock(return_value=streaming_iter),
    ), patch.object(
        router,
        "_aresponses_streaming_iterator",
        new=AsyncMock(return_value=wrapped),
    ) as mock_wrap:
        out = await router._aresponses_with_streaming_fallbacks(
            original_function=fake_original,
            model="primary",
            stream=True,
        )
    assert out is wrapped
    mock_wrap.assert_awaited_once()


# -------- every fallback entry stays reachable across hops --------


def _make_three_tier_router(**router_kwargs) -> Router:
    return Router(
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": "openai/primary-model", "api_key": "sk-test"}},
            {"model_name": "fb1", "litellm_params": {"model": "openai/fb1-model", "api_key": "sk-test"}},
            {"model_name": "fb2", "litellm_params": {"model": "openai/fb2-model", "api_key": "sk-test"}},
        ],
        num_retries=0,
        **router_kwargs,
    )


def _mid_stream_failure(model: str):
    import litellm
    from litellm.exceptions import MidStreamFallbackError

    return MidStreamFallbackError(
        message="stream dropped",
        model=model,
        llm_provider="openai",
        original_exception=litellm.InternalServerError(message="stream dropped", llm_provider="openai", model=model),
        is_pre_first_chunk=True,
    )


def _scripted_responses_stream(events: list, error: Exception | None = None):
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator

    class _ScriptedStream(BaseResponsesAPIStreamingIterator):
        def __init__(self) -> None:
            self._events = list(events)
            self._hidden_params: dict = {}
            self.completed_response = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._events:
                return self._events.pop(0)
            if error is not None:
                raise error
            raise StopAsyncIteration

        async def aclose(self) -> None:
            return None

    return _ScriptedStream()


def _three_tier_original(calls: list, primary_fails_pre_stream: bool):
    import litellm

    completed_event = _make_completed_event(1, 1, 2)

    async def fake_original(**kwargs):
        model = kwargs["model"]
        calls.append(model)
        if model == "openai/primary-model":
            if primary_fails_pre_stream:
                raise litellm.InternalServerError(message="primary down", llm_provider="openai", model=model)
            return _scripted_responses_stream([], _mid_stream_failure(model))
        if model == "openai/fb1-model":
            return _scripted_responses_stream([], _mid_stream_failure(model))
        return _scripted_responses_stream([completed_event])

    return fake_original, completed_event


@pytest.mark.asyncio
async def test_aresponses_pre_stream_primary_failure_then_hop_stream_failure_reaches_second_entry():
    """Regression: fallbacks=[{"primary": ["fb1", "fb2"]}]. The primary fails before streaming,
    fb1 is reached through the regular fallback chain and then fails mid-stream. Only the
    primary's stream used to be wrapped, so fb1's mid-stream failure either re-raised or
    re-tried fb1 itself; fb2 was unreachable."""
    router = _make_three_tier_router(fallbacks=[{"primary": ["fb1", "fb2"]}])
    calls: list = []
    fake_original, completed_event = _three_tier_original(calls, primary_fails_pre_stream=True)

    stream = await router._aresponses_with_streaming_fallbacks(
        original_function=fake_original, model="primary", stream=True, input="hi"
    )
    collected = [event async for event in stream]

    assert calls == ["openai/primary-model", "openai/fb1-model", "openai/fb2-model"]
    assert collected == [completed_event]


@pytest.mark.asyncio
async def test_aresponses_two_consecutive_mid_stream_failures_reach_second_entry():
    """Regression: the primary and fb1 both fail mid-stream; fb2 must still be tried."""
    router = _make_three_tier_router(fallbacks=[{"primary": ["fb1", "fb2"]}])
    calls: list = []
    fake_original, completed_event = _three_tier_original(calls, primary_fails_pre_stream=False)

    stream = await router._aresponses_with_streaming_fallbacks(
        original_function=fake_original, model="primary", stream=True, input="hi"
    )
    collected = [event async for event in stream]

    assert calls == ["openai/primary-model", "openai/fb1-model", "openai/fb2-model"]
    assert collected == [completed_event]


@pytest.mark.asyncio
async def test_aresponses_per_request_fallbacks_survive_into_hop_streams():
    """Regression: a request-level fallbacks list (key or team router_settings) is popped
    before each attempt runs, so a hop's mid-stream re-entry used to see only the router's
    own (empty) list and gave up after fb1."""
    router = _make_three_tier_router()
    calls: list = []
    fake_original, completed_event = _three_tier_original(calls, primary_fails_pre_stream=False)

    stream = await router._aresponses_with_streaming_fallbacks(
        original_function=fake_original,
        model="primary",
        stream=True,
        input="hi",
        fallbacks=[{"primary": ["fb1", "fb2"]}],
    )
    collected = [event async for event in stream]

    assert calls == ["openai/primary-model", "openai/fb1-model", "openai/fb2-model"]
    assert collected == [completed_event]


@pytest.mark.asyncio
async def test_aresponses_attempt_strips_the_controls_carrier_and_wraps_every_hop_stream():
    """Each attempt of the chain, not only the primary's, comes back wrapped for mid-stream
    failover, and the per-request controls carrier rides into the wrapper's re-entry kwargs
    without ever reaching the provider call."""
    from types import MappingProxyType

    from litellm.router_utils.fallback_event_handlers import (
        MID_STREAM_FALLBACK_CONTROLS_KEY,
        MidStreamFallbackControls,
    )

    router = _make_three_tier_router()
    completed_event = _make_completed_event(1, 1, 2)
    hop_stream = _scripted_responses_stream([completed_event])
    seen: dict = {}

    async def fake_original(**kwargs):
        seen.update(kwargs)
        return hop_stream

    controls = MidStreamFallbackControls(MappingProxyType({"fallbacks": [{"primary": ["fb1", "fb2"]}]}))
    stream = await router._ageneric_api_call_with_fallbacks_responses_attempt(
        model="fb1",
        original_generic_function=fake_original,
        stream=True,
        input="hi",
        **{MID_STREAM_FALLBACK_CONTROLS_KEY: controls},
    )
    collected = [event async for event in stream]

    assert seen["model"] == "openai/fb1-model"
    assert MID_STREAM_FALLBACK_CONTROLS_KEY not in seen
    assert "fallbacks" not in seen
    assert stream is not hop_stream
    assert collected == [completed_event]


@pytest.mark.asyncio
async def test_aresponses_fallback_on_in_stream_error_event():
    """A retriable in-stream error event (429) must trigger the router's mid-stream
    fallback path: the wrapper catches MidStreamFallbackError raised by the source
    iterator and yields the fallback stream instead of surfacing the error."""
    import json
    from unittest.mock import Mock

    import litellm
    from litellm.exceptions import MidStreamFallbackError
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
    from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
    from litellm.types.llms.openai import ErrorEvent, ErrorEventError

    router = _make_router()

    error_payload = {
        "type": "error",
        "error": {"type": "tokens", "code": "rate_limit_exceeded", "message": "rate limited"},
    }
    sse_bytes = f"data: {json.dumps(error_payload)}\n\n".encode()

    async def mock_aiter_bytes():
        yield sse_bytes

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_config.transform_streaming_response.return_value = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=0,
        error=ErrorEventError(type="tokens", code="rate_limit_exceeded", message="rate limited"),
    )

    source = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    fallback_event = _make_completed_event(1, 1, 2)

    class _FallbackStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return fallback_event

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_FallbackStream()),
    ) as mock_fallback:
        wrapped = await router._aresponses_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary", "input": "original question"},
        )
        collected = [ev async for ev in wrapped]

    assert collected == [fallback_event]
    mock_fallback.assert_awaited_once()
    raised = mock_fallback.await_args.kwargs["e"]
    assert isinstance(raised, MidStreamFallbackError)
    assert raised.status_code == 429
    assert isinstance(raised.original_exception, litellm.RateLimitError)
    assert raised.original_exception.status_code == 429
    assert mock_fallback.await_args.kwargs["kwargs"]["input"] == "original question"


@pytest.mark.asyncio
async def test_aresponses_fallback_uses_continuation_input_after_partial_content():
    """When output text was already streamed before the error, the fallback re-entry
    must carry a continuation input with the partial assistant text instead of
    retrying the original input from scratch (which would duplicate streamed content)."""
    import json
    from unittest.mock import Mock

    from litellm.exceptions import MidStreamFallbackError
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
    from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
    from litellm.types.llms.openai import ErrorEvent, ErrorEventError

    router = _make_router()

    events = [
        {"type": "response.output_text.delta", "delta": "partial answer"},
        {"type": "error", "error": {"type": "server_error", "code": "internal_error", "message": "boom"}},
    ]
    sse_payload = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)

    async def mock_aiter_bytes():
        yield sse_payload

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    def transform(model, parsed_chunk, logging_obj):
        if parsed_chunk.get("type") == "error":
            return ErrorEvent(
                type=ResponsesAPIStreamEvents.ERROR,
                sequence_number=0,
                error=ErrorEventError(**parsed_chunk["error"]),
            )
        delta_event = Mock()
        delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        delta_event.delta = parsed_chunk["delta"]
        return delta_event

    mock_config.transform_streaming_response.side_effect = transform

    source = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    fallback_event = _make_completed_event(1, 1, 2)

    class _FallbackStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return fallback_event

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_FallbackStream()),
    ) as mock_fallback:
        wrapped = await router._aresponses_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary", "input": "original question"},
        )
        collected = [ev async for ev in wrapped]

    assert collected[-1] == fallback_event
    raised = mock_fallback.await_args.kwargs["e"]
    assert isinstance(raised, MidStreamFallbackError)
    assert raised.is_pre_first_chunk is False
    assert raised.generated_content == "partial answer"
    continuation = mock_fallback.await_args.kwargs["kwargs"]["input"]
    assert isinstance(continuation, list)
    assert continuation[0]["content"][0]["text"] == "original question"
    assert continuation[-2]["role"] == "developer"
    assert continuation[-1]["role"] == "assistant"
    assert continuation[-1]["content"][0]["text"] == "partial answer"


@pytest.mark.asyncio
async def test_aresponses_client_error_event_skips_fallback():
    """A 400-mapped in-stream error (raised as APIError, not MidStreamFallbackError)
    must surface to the caller without invoking the router's fallback path."""
    import litellm

    router = _make_router()

    class _ClientErrorSource:
        completed_response = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise litellm.APIError(
                status_code=400,
                message="bad request",
                llm_provider="openai",
                model="gpt-5",
            )

    wrapped = await router._aresponses_streaming_iterator(
        response=_ClientErrorSource(),
        initial_kwargs={"model": "primary"},
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(),
    ) as mock_fallback:
        with pytest.raises(litellm.APIError) as exc_info:
            async for _ in wrapped:
                pass

    assert exc_info.value.status_code == 400
    mock_fallback.assert_not_awaited()


# -------- non-text output delivered before a mid-stream error must not splice a fallback --------


def _sse_response_replaying(events: tuple[dict[str, Any], ...]):
    """Mock httpx-style response whose byte stream replays `events` as sequence-numbered SSE frames."""
    import json
    from unittest.mock import Mock

    payload = b"".join(
        f"data: {json.dumps({**event, 'sequence_number': index})}\n\n".encode()
        for index, event in enumerate(events)
    )

    async def mock_aiter_bytes() -> AsyncIterator[bytes]:
        yield payload

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    return mock_response


async def _wrapped_sse_stream(router: Router, sse_events: tuple[dict[str, Any], ...], initial_kwargs: dict):
    """Drive Router._aresponses_streaming_iterator over a real SSE byte stream parsed by the real
    OpenAIResponsesAPIConfig, so the guard sees the same typed events production yields."""
    from unittest.mock import MagicMock

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    source = ResponsesAPIStreamingIterator(
        response=_sse_response_replaying(sse_events),
        model="gpt-5",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )
    return await router._aresponses_streaming_iterator(response=source, initial_kwargs=initial_kwargs)


class _SingleEventFallbackStream:
    """Async iterator standing in for the fallback chain's stream: yields exactly one event."""

    def __init__(self, event: object) -> None:
        self._event = event
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration
        self._done = True
        return self._event


_RETRIABLE_SERVER_ERROR_FRAME: dict[str, Any] = {
    "type": "error",
    "error": {"type": "server_error", "code": "server_error", "message": "upstream failed"},
}


@pytest.mark.parametrize(
    "output_event",
    (
        pytest.param(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_first",
                    "call_id": "call_first",
                    "name": "write_todos",
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            id="function-call-added",
        ),
        pytest.param(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_first",
                    "call_id": "call_first",
                    "name": "write_todos",
                    "arguments": "{}",
                    "status": "completed",
                },
            },
            id="function-call-done",
        ),
        pytest.param(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_first", "summary": []},
            },
            id="reasoning-item",
        ),
        pytest.param(
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_first",
                "delta": '{"task":',
            },
            id="function-call-arguments",
        ),
        pytest.param(
            {
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "item_id": "rs_first",
                "summary_index": 0,
                "delta": "Planning the next step",
            },
            id="reasoning-summary",
        ),
        pytest.param(
            {
                "type": "response.content_part.added",
                "output_index": 0,
                "item_id": "msg_first",
                "content_index": 0,
                "part": {"type": "refusal", "refusal": "Cannot comply"},
            },
            id="refusal-part",
        ),
        pytest.param(
            {
                "type": "response.custom_tool_call_input.delta",
                "output_index": 0,
                "item_id": "custom_first",
                "delta": "print(",
            },
            id="custom-tool-input",
        ),
    ),
)
@pytest.mark.parametrize("text_position", ("none", "before", "after"))
@pytest.mark.asyncio
async def test_aresponses_error_after_non_text_output_surfaces_instead_of_splicing(
    output_event: dict[str, Any], text_position: str
):
    """Non-text output (tool calls, reasoning, refusals, argument deltas) already delivered belongs
    to a response the upstream never stored, so a mid-stream error must surface instead of splicing
    a regenerated response: every non-text shape latches the guard, with or without text around it."""
    import litellm

    router = _make_router()
    text_event: dict[str, Any] = {
        "type": "response.output_text.delta",
        "output_index": 1,
        "item_id": "msg_first",
        "content_index": 0,
        "delta": "Let me check",
    }
    partial_events = (
        *((text_event,) if text_position == "before" else ()),
        output_event,
        *((text_event,) if text_position == "after" else ()),
    )
    sse_events = (
        {
            "type": "response.created",
            "response": {"id": "resp_first", "object": "response", "status": "in_progress", "output": []},
        },
        *partial_events,
        _RETRIABLE_SERVER_ERROR_FRAME,
    )

    with patch.object(router, "async_function_with_fallbacks_common_utils", new=AsyncMock()) as mock_fallback:
        wrapped = await _wrapped_sse_stream(router, sse_events, {"model": "primary", "input": "original question"})
        delivered = [await anext(wrapped) for _ in sse_events[:-1]]
        with pytest.raises(litellm.InternalServerError) as exc_info:
            await anext(wrapped)

    mock_fallback.assert_not_awaited()
    assert tuple(event.type for event in delivered) == tuple(event["type"] for event in sse_events[:-1])
    assert exc_info.value.status_code >= 500


@pytest.mark.asyncio
async def test_aresponses_message_shell_with_text_still_falls_back_with_continuation():
    """A message output item and its output_text parts are text output, not non-text: the mid-stream
    error must still take the continuation fallback, whose re-entry carries the partial assistant text."""
    router = _make_router()
    message: dict[str, Any] = {"type": "message", "id": "msg_first", "role": "assistant", "content": []}
    text_part: dict[str, Any] = {"type": "output_text", "text": "partial answer", "annotations": []}
    part_location: dict[str, Any] = {"item_id": "msg_first", "output_index": 0, "content_index": 0}
    sse_events = (
        {
            "type": "response.created",
            "response": {"id": "resp_first", "object": "response", "status": "in_progress", "output": []},
        },
        {"type": "response.output_item.added", "output_index": 0, "item": {**message, "status": "in_progress"}},
        {"type": "response.content_part.added", **part_location, "part": {**text_part, "text": ""}},
        {"type": "response.output_text.delta", **part_location, "delta": "partial answer"},
        {"type": "response.output_text.done", **part_location, "text": "partial answer"},
        {"type": "response.content_part.done", **part_location, "part": text_part},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {**message, "status": "completed", "content": [text_part]},
        },
        _RETRIABLE_SERVER_ERROR_FRAME,
    )
    fallback_event = _make_completed_event(1, 1, 2)

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_SingleEventFallbackStream(fallback_event)),
    ) as mock_fallback:
        collected = [
            event
            async for event in await _wrapped_sse_stream(
                router, sse_events, {"model": "primary", "input": "original question"}
            )
        ]

    mock_fallback.assert_awaited_once()
    assert collected[-1] == fallback_event
    assert tuple(event.type for event in collected[:-1]) == tuple(event["type"] for event in sse_events[:-1])
    continuation = mock_fallback.await_args.kwargs["kwargs"]["input"]
    assert continuation[0]["content"][0]["text"] == "original question"
    assert continuation[-1]["content"][0]["text"] == "partial answer"


@pytest.mark.asyncio
async def test_aresponses_non_text_output_error_without_original_exception_reraises_itself():
    """A MidStreamFallbackError raised by the source iterator with no original exception after
    non-text output re-raises itself: there is no underlying error to surface and no safe splice."""
    from litellm.exceptions import MidStreamFallbackError
    from litellm.types.llms.openai import BaseLiteLLMOpenAIResponseObject, OutputItemAddedEvent

    router = _make_router()
    stream_error = MidStreamFallbackError(
        message="upstream failed",
        model="primary",
        llm_provider="openai",
        original_exception=None,
        generated_content="",
        is_pre_first_chunk=False,
    )
    function_call_event = OutputItemAddedEvent(
        type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        output_index=0,
        item=BaseLiteLLMOpenAIResponseObject(type="function_call", call_id="call_first", name="lookup", arguments=""),
    )

    async def source() -> AsyncIterator[OutputItemAddedEvent]:
        yield function_call_event
        raise stream_error

    with patch.object(router, "async_function_with_fallbacks_common_utils", new=AsyncMock()) as mock_fallback:
        wrapped = await router._aresponses_streaming_iterator(
            response=source(), initial_kwargs={"model": "primary", "input": "original question"}
        )
        assert await anext(wrapped) is function_call_event
        with pytest.raises(MidStreamFallbackError) as exc_info:
            await anext(wrapped)

    assert exc_info.value is stream_error
    mock_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_aresponses_text_partial_still_falls_back_with_continuation():
    """Pure text output keeps the upstream continuation fallback: the error does not surface and the
    fallback re-entry carries the partial assistant text."""
    router = _make_router()
    sse_events = (
        {
            "type": "response.created",
            "response": {"id": "resp_first", "object": "response", "status": "in_progress", "output": []},
        },
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "partial answer"},
        _RETRIABLE_SERVER_ERROR_FRAME,
    )
    fallback_event = _make_completed_event(1, 1, 2)

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_SingleEventFallbackStream(fallback_event)),
    ) as mock_fallback:
        collected = [
            event
            async for event in await _wrapped_sse_stream(
                router, sse_events, {"model": "primary", "input": "original question"}
            )
        ]

    assert collected[-1] == fallback_event
    mock_fallback.assert_awaited_once()
    continuation = mock_fallback.await_args.kwargs["kwargs"]["input"]
    assert continuation[-1]["content"][0]["text"] == "partial answer"


@pytest.mark.asyncio
async def test_aresponses_queued_then_text_still_falls_back():
    """A held response.queued announcement flushed ahead of text output is lifecycle-only, so the
    continuation fallback still runs and the client still sees exactly one completed response."""
    router = _make_router()
    sse_events = (
        {
            "type": "response.queued",
            "response": {"id": "resp_first", "object": "response", "status": "queued", "output": []},
        },
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "partial"},
        _RETRIABLE_SERVER_ERROR_FRAME,
    )
    fallback_event = _make_completed_event(1, 1, 2)

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_SingleEventFallbackStream(fallback_event)),
    ) as mock_fallback:
        collected = [
            event async for event in await _wrapped_sse_stream(router, sse_events, {"model": "primary", "input": "q"})
        ]

    assert collected[-1] == fallback_event
    mock_fallback.assert_awaited_once()


@pytest.mark.asyncio
async def test_aresponses_lifecycle_only_error_still_falls_back():
    """Only held lifecycle events (no output of any kind) keep the plain upstream fallback."""
    router = _make_router()
    sse_events = (
        {
            "type": "response.created",
            "response": {"id": "resp_first", "object": "response", "status": "in_progress", "output": []},
        },
        {
            "type": "response.in_progress",
            "response": {"id": "resp_first", "object": "response", "status": "in_progress", "output": []},
        },
        _RETRIABLE_SERVER_ERROR_FRAME,
    )
    fallback_event = _make_completed_event(1, 1, 2)

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_SingleEventFallbackStream(fallback_event)),
    ) as mock_fallback:
        collected = [
            event async for event in await _wrapped_sse_stream(router, sse_events, {"model": "primary", "input": "q"})
        ]

    assert collected == [fallback_event]
    mock_fallback.assert_awaited_once()
