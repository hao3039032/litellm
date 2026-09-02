from unittest.mock import MagicMock, patch

import httpx

from litellm.llms.runwayml.image_generation.transformation import (
    RunwayMLImageGenerationConfig,
)


def test_runwayml_image_polling_uses_model_proxy():
    proxy = "http://127.0.0.1:8080"
    response = httpx.Response(
        200,
        json={"id": "task-1", "status": "SUCCEEDED", "output": ["https://example.com/image.png"]},
        request=httpx.Request("GET", "https://api.dev.runwayml.com/v1/tasks/task-1"),
    )
    client = MagicMock()
    client.get.return_value = response

    with patch(
        "litellm.llms.custom_httpx.http_handler._get_httpx_client",
        return_value=client,
    ) as get_client:
        result = RunwayMLImageGenerationConfig()._poll_task_sync(
            task_id="task-1",
            api_base="https://api.dev.runwayml.com",
            headers={"Authorization": "Bearer test"},
            litellm_params={"proxy": proxy},
        )

    assert result is response
    assert get_client.call_args.kwargs["params"]["proxy"] == proxy
