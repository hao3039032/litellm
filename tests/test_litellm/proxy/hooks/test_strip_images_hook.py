import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

_MODULE_PATH: Final = Path(__file__).resolve().parents[4] / "docker" / "strip_images.py"
_PLACEHOLDER_MARKER: Final = "does not support image input"


def _load_strip_images(monkeypatch: pytest.MonkeyPatch, configured_models: str | None) -> ModuleType:
    if configured_models is None:
        monkeypatch.delenv("STRIP_IMAGE_MODELS", raising=False)
    else:
        monkeypatch.setenv("STRIP_IMAGE_MODELS", configured_models)
    spec: Final = importlib.util.spec_from_file_location("strip_images_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module: Final = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


async def _run_hook(module: ModuleType, data: dict[str, object]) -> dict[str, object]:
    return await module.proxy_handler_instance.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion"
    )


def _user_message(*blocks: dict[str, object]) -> dict[str, object]:
    return {"role": "user", "content": list(blocks)}


@pytest.mark.asyncio
@pytest.mark.parametrize("request_model", ["glm-5.3", "GLM-5.2", " glm-5.2 "])
async def test_replaces_image_blocks_for_configured_model(monkeypatch: pytest.MonkeyPatch, request_model: str):
    module: Final = _load_strip_images(monkeypatch, "GLM-5.3, glm-5.2")
    data: Final[dict[str, object]] = {
        "model": request_model,
        "messages": [
            _user_message(
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
            )
        ],
    }

    result: Final = await _run_hook(module, data)

    content: Final = result["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert [block["type"] for block in content] == ["text", "text", "text"]
    assert all(_PLACEHOLDER_MARKER in block["text"] for block in content[1:])


@pytest.mark.asyncio
async def test_replaces_images_nested_in_tool_results(monkeypatch: pytest.MonkeyPatch):
    module: Final = _load_strip_images(monkeypatch, "GLM-5.3")
    data: Final[dict[str, object]] = {
        "model": "GLM-5.3",
        "messages": [
            _user_message(
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "text", "text": "screenshot taken"},
                        {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
                    ],
                }
            )
        ],
    }

    result: Final = await _run_hook(module, data)

    tool_result: Final = result["messages"][0]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_1"
    assert tool_result["content"][0] == {"type": "text", "text": "screenshot taken"}
    assert tool_result["content"][1]["type"] == "text"
    assert _PLACEHOLDER_MARKER in tool_result["content"][1]["text"]


@pytest.mark.asyncio
async def test_leaves_unconfigured_models_untouched(monkeypatch: pytest.MonkeyPatch):
    module: Final = _load_strip_images(monkeypatch, "GLM-5.3")
    image_block: Final = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    data: Final[dict[str, object]] = {"model": "gpt-5.4", "messages": [_user_message(image_block)]}

    result: Final = await _run_hook(module, data)

    assert result["messages"][0]["content"] == [image_block]


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_models", [None, ""])
async def test_strips_nothing_when_env_var_is_unset_or_empty(
    monkeypatch: pytest.MonkeyPatch, configured_models: str | None
):
    module: Final = _load_strip_images(monkeypatch, configured_models)
    image_block: Final = {"type": "image", "source": {"type": "base64", "data": "AAAA"}}
    data: Final[dict[str, object]] = {"model": "glm-5.3", "messages": [_user_message(image_block)]}

    result: Final = await _run_hook(module, data)

    assert result["messages"][0]["content"] == [image_block]
