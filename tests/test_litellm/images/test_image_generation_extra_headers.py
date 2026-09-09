"""
Unit test for https://github.com/BerriAI/litellm/issues/22285

Verifies that extra_headers passed to image_generation() are forwarded
to the OpenAI SDK on the openai/litellm_proxy/openai_compatible_providers
code paths.
"""

from unittest.mock import patch

import httpx
import pytest
from openai import AsyncOpenAI

import litellm
from litellm.images.main import image_generation


class TestImageGenerationExtraHeaders:
    """Test that extra_headers are forwarded on the OpenAI code path."""

    @patch("litellm.images.main.openai_chat_completions")
    def test_extra_headers_forwarded_to_openai_image_generation(
        self, mock_openai_chat_completions
    ):
        """
        extra_headers passed to image_generation() should appear in
        optional_params["extra_headers"] when the provider is openai.
        """
        mock_image_response = litellm.utils.ImageResponse(
            created=1234567890,
            data=[{"url": "https://example.com/image.png"}],
        )
        mock_openai_chat_completions.image_generation.return_value = mock_image_response

        extra_headers = {"traceparent": "00-abc123-def456-01", "X-Custom": "value"}

        image_generation(
            model="openai/dall-e-3",
            prompt="A red circle",
            extra_headers=extra_headers,
        )

        mock_openai_chat_completions.image_generation.assert_called_once()
        call_kwargs = mock_openai_chat_completions.image_generation.call_args
        optional_params = call_kwargs.kwargs.get(
            "optional_params", call_kwargs[1].get("optional_params", {})
        )

        assert "extra_headers" in optional_params
        assert optional_params["extra_headers"] == extra_headers

    @patch("litellm.images.main.openai_chat_completions")
    def test_no_extra_headers_when_not_provided(self, mock_openai_chat_completions):
        """
        When extra_headers is not passed, optional_params should not
        contain extra_headers.
        """
        mock_image_response = litellm.utils.ImageResponse(
            created=1234567890,
            data=[{"url": "https://example.com/image.png"}],
        )
        mock_openai_chat_completions.image_generation.return_value = mock_image_response

        image_generation(
            model="openai/dall-e-3",
            prompt="A red circle",
        )

        mock_openai_chat_completions.image_generation.assert_called_once()
        call_kwargs = mock_openai_chat_completions.image_generation.call_args
        optional_params = call_kwargs.kwargs.get(
            "optional_params", call_kwargs[1].get("optional_params", {})
        )

        assert "extra_headers" not in optional_params

    @pytest.mark.asyncio
    async def test_extra_headers_not_serialized_into_request_body(self):
        """
        extra_headers must ride the SDK header transport, never the JSON body.

        Regression test: when the proxy injects data["extra_headers"] (e.g.
        LITELLM_FORWARD_CLIENT_USER_AGENT), it fell into non_default_params,
        landed in extra_body, and the OpenAI SDK merged it into the request
        body as a top-level field, which OpenAI rejects with
        "Unknown parameter: 'extra_headers'".
        """
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content
            captured["headers"] = dict(request.headers)
            return httpx.Response(
                200,
                json={"created": 1234567890, "data": [{"b64_json": "aGk="}]},
            )

        sdk_client = AsyncOpenAI(
            api_key="sk-test",
            base_url=None,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )

        import json

        from litellm.images.main import aimage_generation

        await aimage_generation(
            model="gpt-image-2",
            prompt="A red circle",
            n=1,
            quality="low",
            size="1024x1024",
            extra_headers={"user-agent": "python-requests/2.31"},
            client=sdk_client,
            num_retries=0,
            max_retries=0,
            litellm_call_id="test-call-id",
        )

        body = json.loads(captured["body"])
        assert "extra_headers" not in body
        assert sorted(body.keys()) == ["model", "n", "prompt", "quality", "size"]
        assert "python-requests/2.31" in captured["headers"].get("user-agent", "")

    @pytest.mark.asyncio
    async def test_gpt_image_optional_params_reach_request_body(self):
        """
        Regression test: gpt-image-specific optional params (background,
        output_format, moderation, output_compression) were silently dropped
        because _get_non_default_params only kept params present in the
        hardcoded default_params dict, which lacked these fields. A client
        asking for background=transparent silently got an opaque image.
        """
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content
            return httpx.Response(
                200,
                json={"created": 1234567890, "data": [{"b64_json": "aGk="}]},
            )

        sdk_client = AsyncOpenAI(
            api_key="sk-test",
            base_url=None,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )

        import json

        from litellm.images.main import aimage_generation

        await aimage_generation(
            model="gpt-image-2",
            prompt="A red circle on transparent background",
            n=1,
            quality="low",
            size="1024x1024",
            background="transparent",
            output_format="png",
            client=sdk_client,
            num_retries=0,
            max_retries=0,
            litellm_call_id="test-call-id",
        )

        body = json.loads(captured["body"])
        assert body["background"] == "transparent"
        assert body["output_format"] == "png"
