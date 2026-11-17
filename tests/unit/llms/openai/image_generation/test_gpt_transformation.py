from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.azure.image_generation.gpt_transformation import AzureGPTImageGenerationConfig
from litellm.llms.openai.image_generation.gpt_transformation import GPTImageGenerationConfig
from litellm.types.utils import ImageResponse
from litellm.utils import get_optional_params_image_gen


@pytest.mark.parametrize("config", [GPTImageGenerationConfig(), AzureGPTImageGenerationConfig()])
def test_transform_image_generation_response_keeps_provider_echo(config):
    raw_response = httpx.Response(
        status_code=200,
        json={
            "created": 1788457009,
            "data": [{"b64_json": "/9j/4AAQSkZJRg=="}],
            "output_format": "jpeg",
            "background": "opaque",
            "quality": "low",
            "size": "1024x1024",
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/images/generations"),
    )

    image_response = config.transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={"prompt": "a red apple", "output_format": "jpeg"},
        optional_params={"output_format": "jpeg"},
        litellm_params={},
        encoding=None,
    )

    assert image_response.output_format == "jpeg"
    assert image_response.quality == "low"
    assert image_response.background == "opaque"


@pytest.mark.parametrize("config", [GPTImageGenerationConfig(), AzureGPTImageGenerationConfig()])
def test_gpt_image_26_passthrough_params_are_supported(config):
    supported = config.get_supported_openai_params("gpt-image-2.6")
    assert {"complexity", "frame_count", "frame_duration_ms", "svg"} <= set(supported)


@pytest.mark.parametrize("config", [GPTImageGenerationConfig(), AzureGPTImageGenerationConfig()])
def test_map_openai_params_routes_gpt_image_26_passthrough_into_extra_body(config):
    optional_params = config.map_openai_params(
        non_default_params={
            "complexity": "low",
            "frame_count": 4,
            "frame_duration_ms": 500,
            "svg": True,
            "quality": "high",
        },
        optional_params={},
        model="gpt-image-2.6",
        drop_params=False,
    )

    assert optional_params["quality"] == "high"
    assert optional_params["extra_body"] == {
        "complexity": "low",
        "frame_count": 4,
        "frame_duration_ms": 500,
        "svg": True,
    }


def test_get_optional_params_image_gen_keeps_gpt_image_26_passthrough_in_extra_body():
    """The SDK images.generate call only accepts these through extra_body, so they must survive
    the provider-specific sweep in add_provider_specific_params_to_optional_params"""
    optional_params = get_optional_params_image_gen(
        model="gpt-image-2.6",
        custom_llm_provider="openai",
        provider_config=GPTImageGenerationConfig(),
        drop_params=False,
        complexity="low",
        frame_count=4,
        frame_duration_ms=500,
        svg=True,
        quality="high",
    )

    assert optional_params["quality"] == "high"
    assert optional_params["extra_body"] == {
        "complexity": "low",
        "frame_count": 4,
        "frame_duration_ms": 500,
        "svg": True,
    }
