import os
from typing import Final, Optional, Union

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

_IMAGE_BLOCK_TYPES: Final = frozenset({"image", "image_url"})


def _strip_image_model_ids() -> frozenset[str]:
    configured: Final = os.getenv("STRIP_IMAGE_MODELS", "")
    return frozenset(model_id.strip().lower() for model_id in configured.split(",") if model_id.strip())


_STRIP_IMAGE_MODEL_IDS: Final = _strip_image_model_ids()


def _should_strip_images(model: str) -> bool:
    return model.strip().lower() in _STRIP_IMAGE_MODEL_IDS


def _image_placeholder(model: str) -> str:
    return (
        f"[An image was originally attached here, but it was removed because "
        f"model {model} does not support image input.]"
    )


def _replace_image_blocks(blocks: list, model: str) -> list:
    placeholder: Final = _image_placeholder(model)
    rebuilt: list = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES:
            rebuilt.append({"type": "text", "text": placeholder})
        elif (
            isinstance(block, dict)
            and block.get("type") == "tool_result"
            and isinstance(block.get("content"), list)
        ):
            rebuilt.append({**block, "content": _replace_image_blocks(block["content"], model)})
        else:
            rebuilt.append(block)
    return rebuilt


def _strip_images_from_messages(messages: list, model: str) -> list:
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            message["content"] = _replace_image_blocks(message["content"], model)
    return messages


class StripImagesHandler(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        model: Final = data.get("model") or ""
        if not _should_strip_images(model):
            return data
        messages: Final = data.get("messages")
        if isinstance(messages, list):
            _strip_images_from_messages(messages, model)
        return data


proxy_handler_instance: Final = StripImagesHandler()
