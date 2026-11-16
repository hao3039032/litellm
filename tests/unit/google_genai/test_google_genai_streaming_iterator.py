import asyncio
import inspect
import json
from datetime import datetime
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.google_genai.streaming_iterator import (
    AsyncGoogleGenAIGenerateContentStreamingIterator,
    GoogleGenAIGenerateContentStreamingIterator,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


class _SuccessRecorder(CustomLogger):
    def __init__(self):
        super().__init__()
        self.logged: list[tuple[datetime, object]] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.logged.append((start_time, kwargs.get("completion_start_time")))


def _logging_obj(
    start_time: datetime = datetime(2026, 10, 9, 12, 0, 0), recorder: _SuccessRecorder | None = None
) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="gemini-3.1-flash",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="agenerate_content_stream",
        start_time=start_time,
        litellm_call_id="google-genai-stream",
        function_id="google-genai-stream",
        dynamic_async_success_callbacks=[recorder] if recorder else None,
    )
    logging_obj.update_environment_variables(
        model="gemini-3.1-flash",
        user="",
        optional_params={},
        litellm_params={"custom_llm_provider": "gemini"},
        custom_llm_provider="gemini",
    )
    return logging_obj


@pytest.mark.parametrize(
    "custom_llm_provider, expected_endpoint_type",
    [
        ("gemini", EndpointType.GEMINI),
        ("vertex_ai", EndpointType.VERTEX_AI),
        ("opencode_go", EndpointType.GEMINI),
    ],
)
@pytest.mark.parametrize(
    "iterator_cls",
    [
        AsyncGoogleGenAIGenerateContentStreamingIterator,
        GoogleGenAIGenerateContentStreamingIterator,
    ],
)
def test_streaming_logging_targets_the_provider_that_served_the_request(
    iterator_cls: (
        type[AsyncGoogleGenAIGenerateContentStreamingIterator] | type[GoogleGenAIGenerateContentStreamingIterator]
    ),
    custom_llm_provider: str,
    expected_endpoint_type: EndpointType,
):
    """Routing every google stream through the vertex handler bills gemini/* at vertex_ai/ rates."""
    iterator = iterator_cls(
        response=MagicMock(),
        model="gemini-3.1-flash-image",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider=custom_llm_provider,
    )

    assert iterator.endpoint_type is expected_endpoint_type


def _large_inline_data_event() -> str:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": "A" * 20000,
                            }
                        }
                    ]
                }
            }
        ]
    }
    return f"data: {json.dumps(payload)}"


def _sse_response(body: str) -> httpx.Response:
    return httpx.Response(200, stream=httpx.ByteStream(body.encode()))


@pytest.mark.asyncio
async def test_async_streaming_iterator_yields_complete_sse_events():
    """Large inlineData must not be split across byte-chunk boundaries."""
    iterator = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=_sse_response(_large_inline_data_event() + "\n\n"),
        model="gemini-3.1-flash-image-preview",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = await iterator.__anext__()
    assert chunk.startswith(b"data: ")
    assert chunk.endswith(b"\n\n")
    assert (
        json.loads(chunk[len(b"data: ") : -2])["candidates"][0]["content"]["parts"][0]["inlineData"]["mimeType"]
        == "image/jpeg"
    )


def test_sync_streaming_iterator_yields_complete_sse_events():
    iterator = GoogleGenAIGenerateContentStreamingIterator(
        response=_sse_response(_large_inline_data_event() + "\n\n"),
        model="gemini-3.1-flash-image-preview",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = next(iterator)
    assert chunk.startswith(b"data: ")
    assert chunk.endswith(b"\n\n")
    assert json.loads(chunk[len(b"data: ") : -2])["candidates"][0]["content"]["parts"][0]["inlineData"][
        "data"
    ].startswith("A")


@pytest.mark.asyncio
async def test_async_streaming_iterator_preserves_multi_field_sse_event():
    iterator = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=_sse_response('event: message\ndata: {"text":"hi"}\n\n'),
        model="gemini-test",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = await iterator.__anext__()
    assert chunk == b'event: message\ndata: {"text":"hi"}\n\n'


@pytest.mark.asyncio
async def test_async_streaming_iterator_forwards_sse_comment_events():
    iterator = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=_sse_response(": keepalive\n\n"),
        model="gemini-test",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = await iterator.__anext__()
    assert chunk == b": keepalive\n\n"


_LINE_SEPARATED_TEXT: Final = "before\u2028between\u2029after"


async def _first_chunk(
    iterator: GoogleGenAIGenerateContentStreamingIterator | AsyncGoogleGenAIGenerateContentStreamingIterator,
) -> bytes:
    if isinstance(iterator, AsyncGoogleGenAIGenerateContentStreamingIterator):
        return await iterator.__anext__()
    return next(iterator)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "iterator_cls",
    [AsyncGoogleGenAIGenerateContentStreamingIterator, GoogleGenAIGenerateContentStreamingIterator],
)
async def test_streaming_iterator_keeps_unicode_line_separators_inside_the_event(
    iterator_cls: (
        type[AsyncGoogleGenAIGenerateContentStreamingIterator] | type[GoogleGenAIGenerateContentStreamingIterator]
    ),
) -> None:
    payload: Final = {"candidates": [{"content": {"parts": [{"text": _LINE_SEPARATED_TEXT}]}}]}
    event: Final = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    iterator: Final = iterator_cls(
        response=_sse_response(event),
        model="gemini-3.8-flash",
        logging_obj=_logging_obj(),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk: Final = await _first_chunk(iterator)

    assert chunk == event.encode()
    assert (
        json.loads(chunk[len(b"data: ") : -2])["candidates"][0]["content"]["parts"][0]["text"] == _LINE_SEPARATED_TEXT
    )


def _gemini_event(text: str, finish_reason: str | None = None) -> str:
    candidate: Final = {"content": {"role": "model", "parts": [{"text": text}]}, "index": 0}
    payload: Final = {
        "candidates": [{**candidate, "finishReason": finish_reason} if finish_reason else candidate],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
        "modelVersion": "gemini-3.1-flash",
    }
    return f"data: {json.dumps(payload)}\n\n"


def _exhausted_stream_response() -> httpx.Response:
    return _sse_response(_gemini_event("Hel") + _gemini_event("lo", "STOP"))


def test_sync_iterator_skips_logging_when_streaming_logging_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_streaming_logging", True)
    recorder: Final = _SuccessRecorder()
    iterator: Final = GoogleGenAIGenerateContentStreamingIterator(
        response=_exhausted_stream_response(),
        model="gemini-3.1-flash",
        logging_obj=_logging_obj(recorder=recorder),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunks: Final = tuple(iterator)

    assert b"".join(chunks) == (_gemini_event("Hel") + _gemini_event("lo", "STOP")).encode()
    assert recorder.logged == []


@pytest.mark.asyncio
async def test_async_iterator_skips_logging_when_streaming_logging_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_streaming_logging", True)
    recorder: Final = _SuccessRecorder()
    tasks_before: Final = asyncio.all_tasks()
    iterator: Final = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=_exhausted_stream_response(),
        model="gemini-3.1-flash",
        logging_obj=_logging_obj(recorder=recorder),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunks: Final = [chunk async for chunk in iterator]
    await asyncio.gather(*(asyncio.all_tasks() - tasks_before))

    assert b"".join(chunks) == (_gemini_event("Hel") + _gemini_event("lo", "STOP")).encode()
    assert recorder.logged == []


def test_sync_stream_logging_run_failure_closes_the_coroutine_and_keeps_the_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Final[list[object]] = []  # mutable-ok: test probe captures the logging coroutine

    def failing_run(coroutine: object) -> None:
        captured.append(coroutine)
        raise RuntimeError("no event loop available")

    monkeypatch.setattr(asyncio, "run", failing_run)
    recorder: Final = _SuccessRecorder()
    iterator: Final = GoogleGenAIGenerateContentStreamingIterator(
        response=_exhausted_stream_response(),
        model="gemini-3.1-flash",
        logging_obj=_logging_obj(recorder=recorder),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunks: Final = tuple(iterator)

    assert b"".join(chunks) == (_gemini_event("Hel") + _gemini_event("lo", "STOP")).encode()
    assert recorder.logged == []
    assert len(captured) == 1
    closed_coroutine: Final = captured[0]
    assert inspect.iscoroutine(closed_coroutine)
    assert inspect.getcoroutinestate(closed_coroutine) == inspect.CORO_CLOSED
    with pytest.raises(StopIteration):
        next(iterator)
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_sync_stream_logging_enqueue_failure_closes_the_coroutine_and_keeps_the_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Final[list[object]] = []  # mutable-ok: test probe captures the logging coroutine

    class _BrokenWorker:
        def ensure_initialized_and_enqueue(self, async_coroutine: object) -> None:
            captured.append(async_coroutine)
            raise RuntimeError("logging worker unavailable")

    monkeypatch.setattr(
        "litellm.google_genai.streaming_iterator.GLOBAL_LOGGING_WORKER",
        _BrokenWorker(),
    )
    recorder: Final = _SuccessRecorder()
    iterator: Final = GoogleGenAIGenerateContentStreamingIterator(
        response=_exhausted_stream_response(),
        model="gemini-3.1-flash",
        logging_obj=_logging_obj(recorder=recorder),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunks: Final = tuple(iterator)

    assert b"".join(chunks) == (_gemini_event("Hel") + _gemini_event("lo", "STOP")).encode()
    assert recorder.logged == []
    assert len(captured) == 1
    closed_coroutine: Final = captured[0]
    assert inspect.iscoroutine(closed_coroutine)
    assert inspect.getcoroutinestate(closed_coroutine) == inspect.CORO_CLOSED
    with pytest.raises(StopIteration):
        next(iterator)
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_async_stream_logs_its_first_chunk_time_measured_from_the_call_start():
    call_start: Final = datetime(2026, 10, 9, 12, 0, 0)
    recorder: Final = _SuccessRecorder()
    logging_obj: Final = _logging_obj(call_start, recorder)
    response: Final = httpx.Response(200, content=(_gemini_event("Hel") + _gemini_event("lo", "STOP")).encode())
    iterator: Final = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=response,
        model="gemini-3.1-flash",
        logging_obj=logging_obj,
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )
    tasks_before: Final = asyncio.all_tasks()

    _ = await iterator.__anext__()
    first_chunk_at: Final = logging_obj.completion_start_time
    _ = [chunk async for chunk in iterator]
    await asyncio.gather(*(asyncio.all_tasks() - tasks_before))

    assert first_chunk_at is not None
    assert recorder.logged == [(call_start, first_chunk_at)]
