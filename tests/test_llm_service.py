"""Tests for LLM provider response handling."""

import json

import httpx

from app.services.llm_service import LLMService


def test_municipality_ai_guide_skips_null_content_and_uses_fallback(monkeypatch):
    service = LLMService()
    pool = [
        {"provider": "first", "api_key": "key", "base_url": "https://first.test", "model": "null-model"},
        {"provider": "second", "api_key": "key", "base_url": "https://second.test", "model": "valid-model"},
    ]
    responses = iter([
        {"content": None, "provider": "first", "model": "null-model"},
        {
            "content": json.dumps(
                {
                    "summary": "Guía válida",
                    "alternative_crops": [],
                    "farming_systems": [],
                    "soil_and_fertilizer": [],
                }
            ),
            "provider": "second",
            "model": "valid-model",
        },
    ])

    monkeypatch.setattr(service, "_get_model_pool", lambda: pool)
    monkeypatch.setattr(service, "_select_next_model", lambda _: pool[0])
    monkeypatch.setattr(service, "_record_model_execution", lambda *_: None)
    monkeypatch.setattr(service, "_call_provider", lambda *_args, **_kwargs: next(responses))

    result = service.generate_municipality_ai_guide({"municipality_name": "Test"})

    assert result["status"] == "success"
    assert result["summary"] == "Guía válida"
    assert result["provider"] == "second"


def test_provider_error_detail_reads_json_message():
    response = httpx.Response(400, json={"error": {"message": "Invalid model"}})

    assert LLMService._provider_error_detail(response) == "Invalid model"
