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


def test_explanation_strips_qwen_reasoning_and_uses_final_answer(monkeypatch):
    service = LLMService()
    pool = [{"provider": "groq", "api_key": "key", "base_url": "https://groq.test", "model": "qwen/qwen3.6-27b"}]
    monkeypatch.setattr(service, "_get_model_pool", lambda: pool)
    monkeypatch.setattr(service, "_select_next_model", lambda _: pool[0])
    monkeypatch.setattr(service, "_record_model_execution", lambda *_: None)
    monkeypatch.setattr(
        service,
        "_call_provider",
        lambda *_args, **_kwargs: {
            "content": "<think>Razonamiento interno que no debe mostrarse.</think>\nLa cebolla puede servir en Encino si cuida el drenaje.",
            "provider": "groq",
            "model": "qwen/qwen3.6-27b",
        },
    )

    result = service.generate_explanation({}, "ENCINO")

    assert result["llm_status"] == "success"
    assert result["explanation"] == "La cebolla puede servir en Encino si cuida el drenaje."


def test_explanation_skips_unclosed_reasoning_block(monkeypatch):
    service = LLMService()
    pool = [
        {"provider": "groq", "api_key": "key", "base_url": "https://groq.test", "model": "qwen/qwen3.6-27b"},
        {"provider": "fallback", "api_key": "key", "base_url": "https://fallback.test", "model": "safe-model"},
    ]
    responses = iter([
        {"content": "<think>Razonamiento incompleto", "provider": "groq", "model": "qwen/qwen3.6-27b"},
        {"content": "Respuesta final segura.", "provider": "fallback", "model": "safe-model"},
    ])
    monkeypatch.setattr(service, "_get_model_pool", lambda: pool)
    monkeypatch.setattr(service, "_select_next_model", lambda _: pool[0])
    monkeypatch.setattr(service, "_record_model_execution", lambda *_: None)
    monkeypatch.setattr(service, "_call_provider", lambda *_args, **_kwargs: next(responses))

    result = service.generate_explanation({}, "ENCINO")

    assert result["llm_status"] == "success"
    assert result["explanation"] == "Respuesta final segura."
    assert result["provider"] == "fallback"


def test_provider_error_detail_reads_json_message():
    response = httpx.Response(400, json={"error": {"message": "Invalid model"}})

    assert LLMService._provider_error_detail(response) == "Invalid model"
