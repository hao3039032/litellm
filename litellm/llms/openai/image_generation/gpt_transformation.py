from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object

_JSON_OBJECT: Final = TypeAdapter(dict[str, object], config=ConfigDict(strict=True, hide_input_in_errors=True))

_GPT_IMAGE_EXTRA_BODY_PARAMS: Final[frozenset[str]] = frozenset(
    ("complexity", "frame_count", "frame_duration_ms", "svg")
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.logging import Logging as LiteLLMLoggingObj

    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer


class GPTImageGenerationConfig(BaseImageGenerationConfig):
    """
    OpenAI gpt-image image generation config
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return [
            "background",
            "complexity",
            "frame_count",
            "frame_duration_ms",
            "moderation",
            "n",
            "output_compression",
            "output_format",
            "quality",
            "size",
            "svg",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params: Final = self.get_supported_openai_params(model)
        extra_body: Final[Mapping[str, object]] = {
            k: non_default_params[k] for k in _GPT_IMAGE_EXTRA_BODY_PARAMS if k in non_default_params
        }
        for k in non_default_params:
            if k not in optional_params and k not in extra_body:
                if k in supported_params:
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        if extra_body:
            optional_params["extra_body"] = extra_body  # rebind-ok: same in-place contract as the loop above
        return optional_params

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        response: Final = _JSON_OBJECT.validate_python(raw_response.json())

        stringified_response: Final = response
        ## LOGGING
        logging_obj.post_call(
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=stringified_response,
        )
        image_response: Final[ImageResponse] = convert_to_model_response_object(
            response_object=stringified_response,
            model_response_object=model_response,
            response_type="image_generation",
        )

        # set optional params
        width: Final = optional_params.get("width")
        height: Final = optional_params.get("height")
        requested_size: Final = (
            f"{width}x{height}"
            if isinstance(width, int) and isinstance(height, int)
            else optional_params.get("size", "1024x1024")
        )
        image_response.size = image_response.size or requested_size
        image_response.quality = image_response.quality or optional_params.get("quality", "high")
        image_response.output_format = image_response.output_format or optional_params.get("output_format", "png")

        return image_response
