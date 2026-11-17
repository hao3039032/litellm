import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROMPT: Final = "a red circle on a white background"


class _Image(BaseModel):
    b64_json: str


class _ImageResponse(BaseModel):
    data: tuple[_Image, ...]


def test_image_generation_forwards_gpt_image_26_passthrough_body_fields(gateway: Gateway) -> None:
    """complexity / frame_count / frame_duration_ms / svg must land in the upstream json body"""

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/generations"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": "gpt-image-2.6",
            "prompt": _PROMPT,
            "complexity": "low",
            "frame_count": 4,
            "frame_duration_ms": 500,
            "svg": True,
        }
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": "aW1n"}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-2.6", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = gateway.client.post(
            "/v1/images/generations",
            json={
                "model": model,
                "prompt": _PROMPT,
                "complexity": "low",
                "frame_count": 4,
                "frame_duration_ms": 500,
                "svg": True,
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 200, response.text
        payload: Final = _ImageResponse.model_validate_json(response.content)
        assert [image.b64_json for image in payload.data] == ["aW1n"], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/images/generations")]


@pytest.mark.parametrize(
    "passthrough",
    (
        {"complexity": "low"},
        {"frame_count": 4},
        {"frame_duration_ms": 500},
        {"svg": True},
    ),
)
def test_image_generation_rejects_no_single_gpt_image_26_passthrough_field(
    gateway: Gateway, passthrough: dict[str, JsonValue]
) -> None:
    """Each passthrough field alone must also clear validation, not just the full set"""

    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": "aW1n"}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-2.6", api_base=f"{wire.url}/v1", api_key="synthetic-openai-key"
        )
        response: Final = gateway.client.post(
            "/v1/images/generations",
            json={"model": model, "prompt": _PROMPT, **passthrough},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 200, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/images/generations")]
