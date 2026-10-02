"""
Tests for Z.AI (Zhipu AI) provider - GLM models
"""

import json
import math

import pytest
import respx

import litellm
from litellm import completion
from litellm.cost_calculator import cost_per_token


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


@pytest.fixture
def zai_response():
    """Mock response from Z.AI API"""
    return {
        "id": "chatcmpl-zai-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


def test_get_llm_provider_zai():
    """Test that get_llm_provider correctly identifies zai provider"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider("zai/glm-4.6")
    assert model == "glm-4.6"
    assert provider == "zai"
    assert api_base == "https://api.z.ai/api/paas/v4"


def test_zai_in_provider_lists():
    """Test that zai is registered in all necessary provider lists"""
    assert "zai" in litellm.openai_compatible_providers
    assert "zai" in litellm.provider_list


@pytest.mark.asyncio
async def test_zai_completion_call(respx_mock, zai_response, monkeypatch):
    """Test completion call with zai provider using mocked response"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
        json=zai_response
    )

    response = await litellm.acompletion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert request.method == "POST"
    assert "api.z.ai" in str(request.url)
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == "Bearer test-api-key"


def test_zai_sync_completion(respx_mock, zai_response, monkeypatch):
    """Test synchronous completion call"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
        json=zai_response
    )

    response = completion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25


@pytest.fixture
def zai_thinking_response():
    return {
        "id": "chatcmpl-zai-thinking",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def _captured_body(respx_mock):
    assert len(respx_mock.calls) == 1
    return json.loads(respx_mock.calls[0].request.content.decode("utf-8"))


class TestZaiSupportedParamsWhitelistReasoning:
    """`thinking` and `reasoning_effort` enter the whitelist only when the
    registry marks the model `supports_reasoning: true`. The registry
    update in this PR adds the flag to the entire GLM-4.5 family
    (previously every GLM-4.5 entry in
    `model_prices_and_context_window_backup.json` was missing the flag
    despite docs.z.ai listing GLM-4.5 as the first model with `thinking`
    support), so the gate now unlocks reasoning params for all of them.
    """

    @pytest.fixture(autouse=True)
    def _use_local_model_cost(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")

    @pytest.mark.parametrize(
        "model",
        [
            "glm-4.5",
            "glm-4.5v",
            "glm-4.5-air",
            "glm-4.5-airx",
            "glm-4.5-x",
            "glm-4.5-flash",
            "glm-4.6",
            "glm-4.7",
        ],
    )
    def test_reasoning_params_in_whitelist(self, model):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        params = ZAIChatConfig().get_supported_openai_params(model=model)
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_reasoning_params_excluded_when_registry_flag_missing(self, monkeypatch):
        """Regression guard for the gate. A model whose registry entry
        does NOT mark `supports_reasoning: true` must keep `thinking`
        and `reasoning_effort` OUT of the whitelist — otherwise a new
        ZAI model added to the registry without the flag silently
        accepts reasoning kwargs that the upstream API will reject.
        """
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        # Synthetic model that won't match any registry entry or
        # wildcard pattern.
        params = ZAIChatConfig().get_supported_openai_params(
            model="glm-no-such-future-model-xyz"
        )
        assert "thinking" not in params
        assert "reasoning_effort" not in params


class TestZaiReasoningParamsLandInExtraBody:
    """The OpenAI Python SDK rejects unknown top-level kwargs (e.g.
    `AsyncCompletions.create() got an unexpected keyword argument 'thinking'`),
    so ZAI-specific reasoning fields must travel inside `extra_body` and
    let the SDK flatten them into the HTTP body. Without this wrapping a
    chained-proxy topology (LiteLLM A -> LiteLLM B -> ZAI) drops the
    field on hop 2: hop 1's SDK flattens `extra_body` into a top-level
    `thinking` kwarg, hop 2 re-emits that as a top-level kwarg, and the
    SDK rejects it
    """

    def test_thinking_wraps_into_extra_body(self):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        result = ZAIChatConfig()._map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={},
            model="glm-4.6",
            drop_params=False,
        )
        assert "thinking" not in result
        assert result["extra_body"]["thinking"] == {"type": "disabled"}

    def test_reasoning_effort_wraps_into_extra_body(self):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        result = ZAIChatConfig()._map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="glm-5",
            drop_params=False,
        )
        assert "reasoning_effort" not in result
        assert result["extra_body"]["reasoning_effort"] == "none"

    def test_thinking_merges_into_existing_extra_body(self):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        result = ZAIChatConfig()._map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={"extra_body": {"already_here": True}},
            model="glm-4.6",
            drop_params=False,
        )
        assert result["extra_body"]["already_here"] is True
        assert result["extra_body"]["thinking"] == {"type": "disabled"}

    def test_standard_params_stay_top_level_alongside_thinking(self):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        result = ZAIChatConfig()._map_openai_params(
            non_default_params={
                "max_tokens": 100,
                "temperature": 0.7,
                "thinking": {"type": "enabled"},
            },
            optional_params={},
            model="glm-4.6",
            drop_params=False,
        )
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["extra_body"]["thinking"] == {"type": "enabled"}
        assert "thinking" not in result

    @pytest.mark.asyncio
    async def test_top_level_thinking_kwarg_reaches_http_body(
        self, respx_mock, zai_thinking_response, monkeypatch
    ):
        """Regression for the hop-2 SDK crash. Pre-fix this raised
        `AsyncCompletions.create() got an unexpected keyword argument
        'thinking'`. Post-fix the boundary HTTP body carries `thinking`
        as a top-level field (the OpenAI SDK flattened `extra_body`)
        """
        monkeypatch.setenv("ZAI_API_KEY", "test-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
            json=zai_thinking_response
        )

        await litellm.acompletion(
            model="zai/glm-4.6",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "disabled"},
        )

        body = _captured_body(respx_mock)
        assert body["thinking"] == {"type": "disabled"}
        assert "extra_body" not in body

    @pytest.mark.asyncio
    async def test_extra_body_thinking_reaches_http_body(
        self, respx_mock, zai_thinking_response, monkeypatch
    ):
        monkeypatch.setenv("ZAI_API_KEY", "test-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
            json=zai_thinking_response
        )

        await litellm.acompletion(
            model="zai/glm-4.6",
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"thinking": {"type": "disabled"}},
        )

        body = _captured_body(respx_mock)
        assert body["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_top_level_reasoning_effort_reaches_http_body(
        self, respx_mock, zai_thinking_response, monkeypatch
    ):
        monkeypatch.setenv("ZAI_API_KEY", "test-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
            json=zai_thinking_response
        )

        await litellm.acompletion(
            model="zai/glm-5",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="none",
        )

        body = _captured_body(respx_mock)
        assert body["reasoning_effort"] == "none"

    @pytest.mark.asyncio
    async def test_thinking_works_on_glm_4_5_via_registry_flag(
        self, respx_mock, zai_thinking_response, monkeypatch
    ):
        """End-to-end: GLM-4.5 carries `supports_reasoning: true` in the
        registry, so the gate in `get_supported_openai_params` lets
        `thinking` through and the SDK boundary lands it in the HTTP
        body. Without the registry update this test would fail with the
        SDK rejecting `thinking` as an unsupported param.
        """
        monkeypatch.setenv("ZAI_API_KEY", "test-key")
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
            json=zai_thinking_response
        )

        await litellm.acompletion(
            model="zai/glm-4.5",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "disabled"},
        )

        body = _captured_body(respx_mock)
        assert body["thinking"] == {"type": "disabled"}


class TestGlm5vTurboContextWindow:
    """GLM-5V-Turbo is a GLM-5-series model, not a GLM-4.5v one. Its
    registry entry was seeded by copying the 4.5v values (128k input /
    32k output), which made litellm's pre-call context-window check
    reject a 148,212-token call with `ContextWindowExceededError` even
    though the real model accepts 200k. Per docs.bigmodel.cn the GLM-5
    family (glm-5, glm-5.1, glm-5-turbo, glm-5v-turbo) is uniformly
    200k input / 128k output. These guards pin that so a future
    copy-paste can't silently re-tighten the limit.
    """

    @pytest.fixture(autouse=True)
    def _use_local_model_cost(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")

    def test_input_window_matches_glm5_family(self):
        info = litellm.model_cost["zai/glm-5v-turbo"]
        assert info["max_input_tokens"] == 200000
        assert info["max_output_tokens"] == 128000

    def test_accepts_the_failing_production_payload(self):
        """The live call that surfaced this bug sent 148,212 input
        tokens and was rejected. The registered ceiling must clear it
        with headroom, not trip the pre-call guard.
        """
        info = litellm.model_cost["zai/glm-5v-turbo"]
        assert info["max_input_tokens"] > 148212

    def test_not_the_45v_family_limit(self):
        """Direct guard against the copy-paste source. 128000/32000 are
        the GLM-4.5v values the entry was seeded from; they must never
        return for a GLM-5-series model.
        """
        info = litellm.model_cost["zai/glm-5v-turbo"]
        assert info["max_input_tokens"] != 128000
        assert info["max_output_tokens"] != 32000


class TestGlm45FamilyRegistrySupportsReasoning:
    """docs.z.ai lists GLM-4.5 as the first model family that supports
    `thinking`. The registry had every GLM-4.5 entry marked
    `supports_reasoning: false`, which is the source-of-truth bug that
    let the SDK-kwarg crash hide for so long
    """

    @pytest.mark.parametrize(
        "model_key",
        [
            "zai/glm-4.5",
            "zai/glm-4.5v",
            "zai/glm-4.5-x",
            "zai/glm-4.5-air",
            "zai/glm-4.5-airx",
            "zai/glm-4.5-flash",
        ],
    )
    def test_supports_reasoning_true(self, model_key):
        import os

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        assert model_key in litellm.model_cost
        assert litellm.model_cost[model_key].get("supports_reasoning") is True


class TestZaiWebSearchOptionsMapping:
    """web_search_options must map to ZAI's native web_search chat tool.

    The Anthropic /v1/messages bridge converts a native Anthropic
    ``web_search_*`` server tool into the OpenAI-style ``web_search_options``
    param. ZAI does not understand that param — instead GLM supports a native
    ``{"type": "web_search", "web_search": {...}}`` chat tool. The ZAI config
    must translate between the two.
    """

    def test_web_search_options_in_supported_params(self):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        config = ZAIChatConfig()
        assert "web_search_options" in config.get_supported_openai_params("glm-5.3")

    def test_web_search_options_appends_native_tool(self):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        config = ZAIChatConfig()
        optional_params = {
            "tools": [{"type": "function", "function": {"name": "get_weather"}}]
        }
        result = config._map_openai_params(
            non_default_params={"web_search_options": {}},
            optional_params=optional_params,
            model="glm-5.3",
            drop_params=False,
        )
        assert len(result["tools"]) == 2
        native_tool = result["tools"][1]
        assert native_tool["type"] == "web_search"
        assert native_tool["web_search"]["enable"] is True
        assert native_tool["web_search"]["search_result"] is True
        # existing function tools are untouched
        assert result["tools"][0]["function"]["name"] == "get_weather"

    def test_web_search_options_forwards_known_fields(self):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        config = ZAIChatConfig()
        result = config._map_openai_params(
            non_default_params={
                "web_search_options": {"search_engine": "search_pro", "search_query": "litellm docs"}
            },
            optional_params={},
            model="glm-5.3",
            drop_params=False,
        )
        native = result["tools"][0]["web_search"]
        assert native["search_engine"] == "search_pro"
        assert native["search_query"] == "litellm docs"
