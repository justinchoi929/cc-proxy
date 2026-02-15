import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from main import app

def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "cc-proxy"}

def test_non_streaming_integration():
    """Full round-trip: Anthropic request -> convert -> mock OpenAI response -> convert back."""
    mock_openai_response = {
        "id": "chatcmpl-test",
        "model": "claude-sonnet-4-20250514",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from proxy!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_openai_response

    with patch("main.get_http_client") as mock_client_factory:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_factory.return_value = mock_client

        client = TestClient(app)
        resp = client.post("/v1/messages", json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["text"] == "Hello from proxy!"
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] == 15
    assert body["usage"]["output_tokens"] == 5

def test_upstream_error_integration():
    """Upstream error should be converted to Anthropic error format."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {
        "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
    }

    with patch("main.get_http_client") as mock_client_factory:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_factory.return_value = mock_client

        client = TestClient(app)
        resp = client.post("/v1/messages", json={
            "model": "test",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        })

    assert resp.status_code == 429
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "rate_limit_error"
