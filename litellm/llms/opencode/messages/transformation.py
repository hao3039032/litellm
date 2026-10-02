from collections.abc import Mapping, Sequence
from typing import Final

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import AnthropicMessagesConfig
from litellm.llms.opencode.chat.transformation import OpenCodeGoMessagesChatConfig, OpenCodeZenMessagesChatConfig
from litellm.llms.opencode.common_utils import resolve_opencode_api_key, with_opencode_session_header


class OpenCodeMessagesConfig(AnthropicMessagesConfig):
    def __init__(self, provider: str) -> None:
        self._provider = provider
        self._chat_config = (
            OpenCodeGoMessagesChatConfig() if provider == "opencode_go" else OpenCodeZenMessagesChatConfig()
        )

    @property
    def custom_llm_provider(self) -> str:
        return self._provider

    def should_strip_billing_metadata(self) -> bool:
        return True

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return self._chat_config.get_complete_url(api_base, api_key, model, optional_params, litellm_params, stream)

    def validate_anthropic_messages_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, object], str | None]:  # mutable-ok: Anthropic HTTP handler mutates headers
        base_headers, resolved_base = super().validate_anthropic_messages_environment(
            headers=dict(headers),  # mutable-ok: inherited Anthropic validator requires mutable inputs
            model=model,
            messages=list(messages),  # mutable-ok: inherited Anthropic validator requires mutable inputs
            optional_params=dict(optional_params),  # mutable-ok: inherited Anthropic validator requires mutable inputs
            litellm_params=dict(litellm_params),  # mutable-ok: inherited Anthropic validator requires mutable inputs
            api_key=resolve_opencode_api_key(api_key),
            api_base=api_base,
        )
        stamped_headers: Final = with_opencode_session_header(base_headers, litellm_params)
        return stamped_headers, resolved_base
