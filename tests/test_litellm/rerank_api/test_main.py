import logging
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.llms.together_ai.rerank.handler import TogetherAIRerank
from litellm.llms.together_ai.rerank.transformation import TogetherAIRerankConfig

MARKER_QUERY = "MARKER_QUERY_do_not_log_at_info"
MARKER_DOC = "MARKER_DOC_sensitive_customer_text"


def _mock_cohere_response() -> MagicMock:
    mock_response = MagicMock()

    def return_val():
        return {
            "id": "cmpl-mockid",
            "results": [{"index": 0, "relevance_score": 0.95}],
            "meta": {
                "api_version": {"version": "1.0"},
                "billed_units": {"search_units": 1},
            },
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200
    return mock_response


def test_rerank_does_not_log_request_content_at_info(caplog):
    """Regression for #32525: rerank must not emit query/documents to logs at INFO.

    The mapped ``optional_rerank_params`` (which always contains ``query`` and
    ``documents``) bypasses ``turn_off_message_logging`` / ``redact_messages``,
    so logging it at INFO leaks raw request content into stdout and any log sink.
    """
    litellm.cohere_key = "test_api_key"
    caplog.set_level(logging.DEBUG, logger="LiteLLM")

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=_mock_cohere_response(),
    ):
        litellm.rerank(
            model="cohere/rerank-english-v3.0",
            query=MARKER_QUERY,
            documents=[MARKER_DOC, "unrelated"],
            top_n=2,
        )

    litellm_records = [r for r in caplog.records if r.name == "LiteLLM"]

    info_or_above = [
        r.getMessage()
        for r in litellm_records
        if r.levelno >= logging.INFO and (MARKER_QUERY in r.getMessage() or MARKER_DOC in r.getMessage())
    ]
    assert not info_or_above, f"rerank leaked request content at INFO+: {info_or_above}"

    optional_params_logs = [r for r in litellm_records if "optional_rerank_params" in r.getMessage()]
    assert optional_params_logs, "expected the optional_rerank_params line to be logged"
    assert all(r.levelno == logging.DEBUG for r in optional_params_logs), (
        "optional_rerank_params must be logged at DEBUG, not INFO"
    )


def test_together_rerank_handler_uses_model_proxy():
    proxy = "http://127.0.0.1:8080"
    client = MagicMock()
    client.post.return_value.status_code = 200
    client.post.return_value.json.return_value = {}

    with (
        patch(
            "litellm.llms.together_ai.rerank.handler._get_httpx_client",
            return_value=client,
        ) as get_client,
        patch.object(TogetherAIRerankConfig, "_transform_response", return_value=MagicMock()),
    ):
        TogetherAIRerank().rerank(
            model="mxbai-rerank-large-v1",
            api_key="test-key",
            query="hello",
            documents=["world"],
            litellm_params={"proxy": proxy},
        )

    assert get_client.call_args.kwargs["params"]["proxy"] == proxy


def test_together_rerank_passes_model_proxy_to_handler():
    proxy = "http://127.0.0.1:8080"

    with patch(
        "litellm.rerank_api.main.together_rerank.rerank",
        return_value=MagicMock(),
    ) as rerank:
        litellm.rerank(
            model="together_ai/mxbai-rerank-large-v1",
            api_key="test-key",
            query="hello",
            documents=["world"],
            proxy=proxy,
        )

    assert rerank.call_args.kwargs["litellm_params"]["proxy"] == proxy
