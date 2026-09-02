"""
Test RunwayML text-to-speech transformation
"""

import os
import sys
from unittest.mock import MagicMock, patch

import httpx

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.runwayml.text_to_speech.transformation import (
    RunwayMLTextToSpeechConfig,
)


def test_openai_voice_mapping_to_runwayml():
    """
    Test that OpenAI voice names are correctly mapped to RunwayML preset IDs
    """
    config = RunwayMLTextToSpeechConfig()

    # Test OpenAI voice mappings
    openai_to_runway = {
        "alloy": "Maya",
        "echo": "James",
        "fable": "Bernard",
        "onyx": "Vincent",
        "nova": "Serene",
        "shimmer": "Ella",
    }

    for openai_voice, expected_runway_voice in openai_to_runway.items():
        mapped_voice, mapped_params = config.map_openai_params(
            model="eleven_multilingual_v2",
            optional_params={},
            voice=openai_voice,
            drop_params=False,
            kwargs={},
        )

        assert mapped_voice is None
        assert "runwayml_voice" in mapped_params
        assert mapped_params["runwayml_voice"]["type"] == "runway-preset"
        assert mapped_params["runwayml_voice"]["presetId"] == expected_runway_voice


def test_runwayml_native_voice_passthrough():
    """
    Test that RunwayML native voice names are passed through correctly as-is
    """
    config = RunwayMLTextToSpeechConfig()

    # Test various RunwayML native voices
    runway_voices = ["Bernard", "Maya", "Arjun", "Serene", "Chad"]

    for runway_voice in runway_voices:
        mapped_voice, mapped_params = config.map_openai_params(
            model="eleven_multilingual_v2",
            optional_params={},
            voice=runway_voice,
            drop_params=False,
            kwargs={},
        )

        assert mapped_voice is None
        assert "runwayml_voice" in mapped_params
        assert mapped_params["runwayml_voice"]["type"] == "runway-preset"
        assert mapped_params["runwayml_voice"]["presetId"] == runway_voice


def test_runwayml_tts_follow_up_requests_use_model_proxy():
    proxy = "http://127.0.0.1:8080"
    raw_response = httpx.Response(
        200,
        json={"id": "task-1", "status": "PENDING"},
        request=httpx.Request(
            "POST",
            "https://api.dev.runwayml.com/v1/text_to_speech",
            headers={"Authorization": "Bearer test", "X-Runway-Version": "2024-11-06"},
        ),
    )
    polled_response = httpx.Response(
        200,
        json={"id": "task-1", "status": "SUCCEEDED", "output": ["https://example.com/audio.mp3"]},
        request=httpx.Request("GET", "https://api.dev.runwayml.com/v1/tasks/task-1"),
    )
    audio_response = httpx.Response(
        200,
        content=b"audio",
        request=httpx.Request("GET", "https://example.com/audio.mp3"),
    )
    poll_client = MagicMock()
    poll_client.get.return_value = polled_response
    download_client = MagicMock()
    download_client.get.return_value = audio_response

    with patch(
        "litellm.llms.custom_httpx.http_handler._get_httpx_client",
        side_effect=[poll_client, download_client],
    ) as get_client:
        RunwayMLTextToSpeechConfig().transform_text_to_speech_response(
            model="eleven_multilingual_v2",
            raw_response=raw_response,
            logging_obj=MagicMock(),
            litellm_params={"proxy": proxy},
        )

    assert get_client.call_count == 2
    assert all(call.kwargs["params"]["proxy"] == proxy for call in get_client.call_args_list)
