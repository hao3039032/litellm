import json
from pathlib import Path

import pytest

import litellm
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config

GPT6_MODELS = ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"]


def _load_model_cost() -> dict:
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_gpt6_model_info_present(model):
    model_cost = _load_model_cost()
    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "openai"
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 922000
    assert info["max_output_tokens"] == 128000
    assert info["max_tokens"] == 128000


def test_gpt6_luna_pricing_matches_openai_pricing_page():
    """gpt-6-luna: $0.10 in / $0.01 cached read / $0.125 cache write / $0.50 out per 1M,
    2x in/cache + 1.5x out above 272K, flex at 50%, fast (priority) at 2x."""
    info = _load_model_cost()["gpt-6-luna"]
    assert info["input_cost_per_token"] == 1e-07
    assert info["output_cost_per_token"] == 5e-07
    assert info["cache_read_input_token_cost"] == 1e-08
    assert info["cache_creation_input_token_cost"] == 1.25e-07

    assert info["input_cost_per_token_above_272k_tokens"] == 2e-07
    assert info["output_cost_per_token_above_272k_tokens"] == 7.5e-07
    assert info["cache_read_input_token_cost_above_272k_tokens"] == 2e-08
    assert info["cache_creation_input_token_cost_above_272k_tokens"] == 2.5e-07

    assert info["input_cost_per_token_flex"] == 5e-08
    assert info["output_cost_per_token_flex"] == 2.5e-07
    assert info["input_cost_per_token_priority"] == 2e-07
    assert info["output_cost_per_token_priority"] == 1e-06
    assert info["input_cost_per_token_batches"] == 5e-08
    assert info["output_cost_per_token_batches"] == 2.5e-07


def test_gpt6_reasoning_effort_capabilities_match_openai_docs():
    """gpt-6-sol/luna document none/low/medium(default)/high/xhigh/max; gpt-6-astra
    documents low/medium/high/xhigh/max and takes no none."""
    info = _load_model_cost()
    for model in ("gpt-6-sol", "gpt-6-luna"):
        assert info[model]["supports_reasoning"] is True
        assert info[model]["supports_none_reasoning_effort"] is True
        assert info[model]["supports_minimal_reasoning_effort"] is False
        assert info[model]["supports_xhigh_reasoning_effort"] is True
        assert info[model]["supports_max_reasoning_effort"] is True

    assert info["gpt-6-astra"]["supports_reasoning"] is True
    assert info["gpt-6-astra"]["supports_none_reasoning_effort"] is False
    assert info["gpt-6-astra"]["supports_max_reasoning_effort"] is True


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_gpt6_models_advertise_reasoning_params(model):
    """Without the gpt-6 -> GPT-5 config path, reasoning_effort/verbosity are absent
    from the supported params and litellm rejects them before the request leaves."""
    supported = litellm.OpenAIConfig().get_supported_openai_params(model)
    assert "reasoning_effort" in supported
    assert "verbosity" in supported


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_gpt6_maps_max_tokens_to_max_completion_tokens(model):
    """OpenAI rejects raw max_tokens for gpt-6 with a 400; the GPT-5 config path
    rewrites it. This is the regression test for the health-check probe (max_tokens=16)
    and for clients that still send max_tokens."""
    optional_params: dict = {}
    non_default_params = {"max_tokens": 50, "reasoning_effort": "low"}
    result = litellm.OpenAIConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert result["max_completion_tokens"] == 50
    assert "max_tokens" not in result
    assert result["reasoning_effort"] == "low"


def test_gpt6_rides_the_responses_bridge_gate():
    """main.py gates the automatic /v1/responses bridge (tools + reasoning_effort) on
    is_model_gpt_5_4_plus_model; gpt-6 must clear it like gpt-5.6 does."""
    assert OpenAIGPT5Config.is_model_gpt_5_4_plus_model("gpt-6-luna") is True
    assert OpenAIGPT5Config.is_model_gpt_5_4_plus_model("gpt-6-astra") is True
    assert OpenAIGPT5Config.is_model_gpt_5_4_plus_model("gpt-5.3") is False
