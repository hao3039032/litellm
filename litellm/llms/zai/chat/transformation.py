from typing import Final

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE: Final = "https://api.z.ai/api/paas/v4"

_REASONING_PARAMS = ("thinking", "reasoning_effort")

_ZAI_NATIVE_WEB_SEARCH_TOOL: Final = "web_search"


class ZAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "zai"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("ZAI_API_BASE") or ZAI_API_BASE
        dynamic_api_key: Final = api_key or get_secret_str("ZAI_API_KEY")
        return api_base, dynamic_api_key

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> tuple[list[AllMessageValues], list[ChatCompletionToolParam] | None]:
        """
        Override to preserve cache_control for GLM/ZAI.
        GLM supports cache_control - don't strip it.
        """
        # GLM/ZAI supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        # ZAI is OpenAI-compatible, so start from the full OpenAI supported
        # params (frequency_penalty, parallel_tool_calls, response_format,
        # seed, n, web_search_options, ...) and add reasoning params for
        # reasoning-capable models.
        base_params: Final = super().get_supported_openai_params(model)

        import litellm

        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider=self.custom_llm_provider):
                base_params.extend(_REASONING_PARAMS)
        except Exception:
            pass

        return base_params

    def _map_web_search_options_to_native_tool(self, web_search_options: dict) -> dict:
        """
        Map the OpenAI-style ``web_search_options`` param to ZAI's native
        ``{"type": "web_search", "web_search": {...}}`` chat tool.

        ZAI native fields (per bigmodel.cn docs):
        - enable (bool): turn the tool on
        - search_query (str, optional): force a custom search query
        - search_result (bool): return per-source search details
        OpenAI ``web_search_options`` may carry:
        - search_context_size ("low"/"medium"/"high")
        - user_location
        Only fields ZAI understands are forwarded; unknown ones are ignored.
        """
        native: dict = {"enable": True, "search_result": True}
        if isinstance(web_search_options, dict):
            forced_query: Final = web_search_options.get("search_query")
            if isinstance(forced_query, str) and forced_query:
                native["search_query"] = forced_query
            search_engine: Final = web_search_options.get("search_engine")
            if isinstance(search_engine, str) and search_engine:
                native["search_engine"] = search_engine
        return {"type": _ZAI_NATIVE_WEB_SEARCH_TOOL, _ZAI_NATIVE_WEB_SEARCH_TOOL: native}

    def _map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param not in supported:
                continue
            if param == "web_search_options":
                # Translate to ZAI's native web_search chat tool and append it
                # to the tools list instead of forwarding an unknown param.
                tools_list: list = optional_params.setdefault("tools", [])
                tools_list.append(self._map_web_search_options_to_native_tool(value or {}))
            elif param in _REASONING_PARAMS:
                # Responses API may pass {"effort": "high"}; extract "effort" field
                if isinstance(value, dict):
                    value = value.get("effort", value)
                optional_params.setdefault("extra_body", {})[param] = value
            else:
                optional_params[param] = value
        return optional_params
