import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.llms.openai.common_utils import BaseOpenAILLM
from litellm.llms.openai.openai import OpenAIChatCompletion, OpenAIError
from litellm.types.utils import EmbeddingResponse

# Test parameters for different API functions
API_FUNCTION_PARAMS = [
    # (function_name, is_async, args)
    (
        "completion",
        False,
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        },
    ),
    (
        "completion",
        True,
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        },
    ),
    (
        "completion",
        True,
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": True,
        },
    ),
    ("embedding", False, {"model": "text-embedding-ada-002", "input": "Hello world"}),
    ("embedding", True, {"model": "text-embedding-ada-002", "input": "Hello world"}),
    (
        "image_generation",
        False,
        {"model": "dall-e-3", "prompt": "A beautiful sunset over mountains"},
    ),
    (
        "image_generation",
        True,
        {"model": "dall-e-3", "prompt": "A beautiful sunset over mountains"},
    ),
    (
        "speech",
        False,
        {
            "model": "tts-1",
            "input": "Hello, this is a test of text to speech",
            "voice": "alloy",
        },
    ),
    (
        "speech",
        True,
        {
            "model": "tts-1",
            "input": "Hello, this is a test of text to speech",
            "voice": "alloy",
        },
    ),
    ("transcription", False, {"model": "whisper-1", "file": MagicMock()}),
    ("transcription", True, {"model": "whisper-1", "file": MagicMock()}),
]


@pytest.mark.parametrize("function_name,is_async,args", API_FUNCTION_PARAMS)
@pytest.mark.asyncio
async def test_openai_client_reuse(function_name, is_async, args):
    """
    Test that multiple API calls reuse the same OpenAI client
    """
    litellm.set_verbose = True

    # Determine which client class to mock based on whether the test is async
    client_path = "litellm.llms.openai.openai.AsyncOpenAI" if is_async else "litellm.llms.openai.openai.OpenAI"

    # Create the appropriate patches
    with (
        patch(client_path) as mock_client_class,
        patch.object(BaseOpenAILLM, "set_cached_openai_client") as mock_set_cache,
        patch.object(BaseOpenAILLM, "get_cached_openai_client") as mock_get_cache,
    ):
        # Setup the mock to return None first time (cache miss) then a client for subsequent calls
        mock_client = MagicMock()
        mock_get_cache.side_effect = [None] + [mock_client] * 9  # First call returns None, rest return the mock client

        # Make 10 API calls
        for _ in range(10):
            try:
                # Call the appropriate function based on parameters
                if is_async:
                    # Add 'a' prefix for async functions
                    func = getattr(litellm, f"a{function_name}")
                    await func(**args)
                else:
                    func = getattr(litellm, function_name)
                    func(**args)
            except Exception:
                # We expect exceptions since we're mocking the client
                pass

        # Verify client was created only once
        assert mock_client_class.call_count == 1, (
            f"{'Async' if is_async else ''}OpenAI client should be created only once"
        )

        # Verify the client was cached
        assert mock_set_cache.call_count == 1, "Client should be cached once"

        # Verify we tried to get from cache 10 times (once per request)
        assert mock_get_cache.call_count == 10, "Should check cache for each request"


def test_precomputed_init_params_match_inspect_signature():
    """
    Verify that the pre-computed _OPENAI_INIT_PARAMS and _AZURE_OPENAI_INIT_PARAMS
    match what inspect.signature() returns. If the OpenAI SDK changes its __init__
    params, this test will fail — signaling the constants need updating.
    """
    import inspect

    from openai import AzureOpenAI, OpenAI

    from litellm.llms.openai.common_utils import (
        _AZURE_OPENAI_INIT_PARAMS,
        _OPENAI_INIT_PARAMS,
    )

    expected_openai = tuple(p for p in inspect.signature(OpenAI.__init__).parameters if p != "self")
    expected_azure = tuple(p for p in inspect.signature(AzureOpenAI.__init__).parameters if p != "self")

    assert _OPENAI_INIT_PARAMS == expected_openai
    assert _AZURE_OPENAI_INIT_PARAMS == expected_azure


@pytest.mark.parametrize("client_type", ["openai", "azure"])
def test_get_openai_client_initialization_param_fields(client_type):
    """Verify the method returns the correct pre-computed params for each client type."""
    result = BaseOpenAILLM.get_openai_client_initialization_param_fields(client_type)
    assert isinstance(result, tuple)
    assert len(result) > 0
    assert "self" not in result


@pytest.mark.parametrize("client_type", ["openai", "azure"])
def test_get_openai_client_cache_key(client_type):
    """Verify get_openai_client_cache_key doesn't raise on tuple + tuple concatenation."""
    key = BaseOpenAILLM.get_openai_client_cache_key(
        client_initialization_params={"api_key": "sk-test"},
        client_type=client_type,
    )
    assert isinstance(key, str)
    assert "api_key=sk-test" in key


def test_openai_client_cache_key_hashes_proxy_credentials():
    proxy = "socks5h://proxy-user:proxy-password@127.0.0.1:1080"
    key = BaseOpenAILLM.get_openai_client_cache_key(
        client_initialization_params={"api_key": "sk-test", "proxy": proxy},
        client_type="openai",
    )

    assert proxy not in key
    assert "proxy-user" not in key
    assert "proxy-password" not in key
    assert "proxy_fingerprint=" in key


def test_openai_client_uses_model_proxy():
    proxy = "http://127.0.0.1:8080"
    openai_handler = OpenAIChatCompletion()

    with (
        patch.object(openai_handler, "get_cached_openai_client", return_value=None),
        patch.object(openai_handler, "set_cached_openai_client"),
        patch.object(OpenAIChatCompletion, "_get_sync_http_client") as get_http_client,
        patch("litellm.llms.openai.openai.OpenAI") as openai_client,
    ):
        openai_handler._get_openai_client(
            is_async=False,
            api_key="sk-test",
            proxy=proxy,
        )

    get_http_client.assert_called_once_with(proxy=proxy)
    assert openai_client.call_args.kwargs["http_client"] is get_http_client.return_value


def test_explicit_openai_client_takes_precedence_over_model_proxy():
    proxy = "http://127.0.0.1:8080"
    provided_client = MagicMock()
    openai_handler = OpenAIChatCompletion()

    with patch.object(OpenAIChatCompletion, "_get_sync_http_client") as get_http_client:
        result = openai_handler._get_openai_client(
            is_async=False,
            client=provided_client,
            proxy=proxy,
        )

    assert result is provided_client
    get_http_client.assert_not_called()


def test_completion_passes_proxy_only_in_provider_litellm_params():
    proxy = "socks5h://proxy-user:proxy-password@127.0.0.1:1080"
    response = litellm.ModelResponse()

    with patch.object(OpenAIChatCompletion, "completion", return_value=response) as completion:
        result = litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-test",
            proxy=proxy,
        )

    assert result is response
    assert completion.call_args.kwargs["litellm_params"]["proxy"] == proxy
    logging_obj = completion.call_args.kwargs["logging_obj"]
    assert "proxy" not in logging_obj.model_call_details["litellm_params"]


def test_embedding_passes_proxy_to_openai_handler():
    proxy = "socks5h://proxy-user:proxy-password@127.0.0.1:1080"
    response = EmbeddingResponse()

    with patch.object(OpenAIChatCompletion, "embedding", return_value=response) as embedding:
        result = litellm.embedding(
            model="openai/text-embedding-3-small",
            input=["hello"],
            api_key="sk-test",
            proxy=proxy,
        )

    assert result is response
    assert embedding.call_args.kwargs["proxy"] == proxy
    logging_obj = embedding.call_args.kwargs["logging_obj"]
    assert "proxy" not in logging_obj.model_call_details["litellm_params"]


def test_openai_embedding_uses_model_proxy():
    proxy = "http://127.0.0.1:8080"
    openai_handler = OpenAIChatCompletion()

    with (
        patch.object(openai_handler, "_get_openai_client", side_effect=RuntimeError("stop")) as get_client,
        pytest.raises(OpenAIError),
    ):
        openai_handler.embedding(
            model="text-embedding-3-small",
            input=["hello"],
            timeout=30,
            logging_obj=MagicMock(),
            model_response=EmbeddingResponse(),
            optional_params={},
            api_key="sk-test",
            proxy=proxy,
        )

    assert get_client.call_args.kwargs["proxy"] == proxy


@pytest.mark.asyncio
async def test_async_openai_embedding_uses_model_proxy():
    proxy = "http://127.0.0.1:8080"
    openai_handler = OpenAIChatCompletion()

    with (
        patch.object(openai_handler, "_get_openai_client", side_effect=RuntimeError("stop")) as get_client,
        pytest.raises(OpenAIError),
    ):
        await openai_handler.aembedding(
            input=["hello"],
            data={"model": "text-embedding-3-small", "input": ["hello"]},
            model_response=EmbeddingResponse(),
            timeout=30,
            logging_obj=MagicMock(),
            api_key="sk-test",
            proxy=proxy,
        )

    assert get_client.call_args.kwargs["proxy"] == proxy


def test_speech_passes_proxy_to_openai_handler():
    proxy = "socks5h://proxy-user:proxy-password@127.0.0.1:1080"

    with patch.object(OpenAIChatCompletion, "audio_speech", return_value=MagicMock()) as audio_speech:
        litellm.speech(
            model="openai/tts-1",
            input="hello",
            voice="alloy",
            api_key="sk-test",
            proxy=proxy,
        )

    assert audio_speech.call_args.kwargs["proxy"] == proxy


def test_openai_audio_speech_uses_model_proxy():
    proxy = "http://127.0.0.1:8080"
    openai_handler = OpenAIChatCompletion()

    with (
        patch.object(openai_handler, "_get_openai_client", side_effect=RuntimeError("stop")) as get_client,
        pytest.raises(RuntimeError),
    ):
        openai_handler.audio_speech(
            model="tts-1",
            input="hello",
            voice="alloy",
            optional_params={},
            api_key="sk-test",
            api_base=None,
            organization=None,
            project=None,
            max_retries=0,
            timeout=30,
            proxy=proxy,
        )

    assert get_client.call_args.kwargs["proxy"] == proxy


@pytest.mark.asyncio
async def test_async_openai_audio_speech_uses_model_proxy():
    proxy = "http://127.0.0.1:8080"
    openai_handler = OpenAIChatCompletion()

    with (
        patch.object(openai_handler, "_get_openai_client", side_effect=RuntimeError("stop")) as get_client,
        pytest.raises(RuntimeError),
    ):
        await openai_handler.async_audio_speech(
            model="tts-1",
            input="hello",
            voice="alloy",
            optional_params={},
            api_key="sk-test",
            api_base=None,
            organization=None,
            project=None,
            max_retries=0,
            timeout=30,
            proxy=proxy,
        )

    assert get_client.call_args.kwargs["proxy"] == proxy


def test_openai_audio_transcription_uses_model_proxy():
    from litellm.llms.openai.transcriptions.handler import OpenAIAudioTranscription
    from litellm.types.utils import TranscriptionResponse

    proxy = "http://127.0.0.1:8080"
    handler = OpenAIAudioTranscription()

    with (
        patch.object(handler, "_get_openai_client", side_effect=RuntimeError("stop")) as get_client,
        pytest.raises(RuntimeError),
    ):
        handler.audio_transcriptions(
            model="whisper-1",
            audio_file=b"audio-bytes",
            optional_params={},
            litellm_params={"proxy": proxy},
            model_response=TranscriptionResponse(),
            timeout=30,
            max_retries=0,
            logging_obj=MagicMock(),
            api_key="sk-test",
            api_base=None,
        )

    assert get_client.call_args.kwargs["proxy"] == proxy


@pytest.mark.asyncio
async def test_async_openai_audio_transcription_uses_model_proxy():
    from litellm.llms.openai.transcriptions.handler import OpenAIAudioTranscription
    from litellm.types.utils import TranscriptionResponse

    proxy = "http://127.0.0.1:8080"
    handler = OpenAIAudioTranscription()

    with (
        patch.object(handler, "_get_openai_client", side_effect=RuntimeError("stop")) as get_client,
        pytest.raises(RuntimeError),
    ):
        await handler.async_audio_transcriptions(
            audio_file=b"audio-bytes",
            data={"model": "whisper-1", "file": b"audio-bytes"},
            model_response=TranscriptionResponse(),
            timeout=30,
            logging_obj=MagicMock(),
            api_key="sk-test",
            proxy=proxy,
        )

    assert get_client.call_args.kwargs["proxy"] == proxy
