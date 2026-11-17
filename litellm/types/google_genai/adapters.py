from collections.abc import Mapping
from typing import TYPE_CHECKING

from openai.types.chat import ChatCompletionStreamOptionsParam
from typing_extensions import ReadOnly, TypedDict

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionToolChoiceStringValues,
    ChatCompletionToolParam,
)

if TYPE_CHECKING:
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession


class GenerateContentCompletionKwargs(TypedDict, total=False):
    model: str
    messages: list[AllMessageValues]
    temperature: float
    max_tokens: int
    top_p: float
    stop: str | list[str]
    tools: list[ChatCompletionToolParam]
    tool_choice: ChatCompletionToolChoiceStringValues
    stream: bool
    metadata: dict[str, object]
    litellm_metadata: ReadOnly[Mapping[str, object]]
    stream_options: ReadOnly[ChatCompletionStreamOptionsParam]
    extra_headers: dict[str, str] | None
    proxy_server_request: dict[str, object] | None
    github_copilot_user_session: ReadOnly["GithubCopilotUserSession"]
