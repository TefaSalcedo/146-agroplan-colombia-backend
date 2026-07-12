"""LLM service for generating explanations and validating prediction results.

Supports OpenRouter and Groq with automatic provider fallback.
When all LLM providers fail, returns None and the caller returns a 200
with data intact and explanation fields set to null.
"""

import json
import time
from typing import Optional, Dict, Any, List

import httpx

from app.config import get_settings

settings = get_settings()

PROMPT_SCHEMA_VERSION = "1.0"


class LLMService:
    """LLM service with multi-provider support and fallback."""

    def __init__(self):
        self.timeout = settings.llm_timeout_seconds

    def _get_providers(self) -> List[Dict[str, Any]]:
        """Build ordered list of providers to try."""
        providers = []

        primary = settings.llm_provider.lower()

        if primary == "openrouter":
            providers.extend(self._openrouter_config())
            providers.extend(self._groq_config())
        else:
            providers.extend(self._groq_config())
            providers.extend(self._openrouter_config())

        return providers

    def _openrouter_config(self) -> List[Dict[str, Any]]:
        models = settings.openrouter_models_list
        if not settings.openrouter_api_key or not models:
            return []
        return [
            {
                "provider": "openrouter",
                "api_key": settings.openrouter_api_key,
                "models": models,
                "base_url": "https://openrouter.ai/api/v1",
            }
        ]

    def _groq_config(self) -> List[Dict[str, Any]]:
        models = settings.groq_models_list
        if not settings.groq_api_key or not models:
            return []
        return [
            {
                "provider": "groq",
                "api_key": settings.groq_api_key,
                "models": models,
                "base_url": "https://api.groq.com/openai/v1",
            }
        ]

    def _build_chat_request(
        self,
        provider: Dict[str, Any],
        model: str,
        system_prompt: str,
        user_content: str,
        response_format: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Build an OpenAI-compatible chat completion request."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        }

        if response_format:
            payload["response_format"] = response_format

        return payload

    def _call_provider(
        self,
        provider: Dict[str, Any],
        model: str,
        system_prompt: str,
        user_content: str,
        response_format: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Call a single provider/model. Returns parsed response or None."""
        base_url = provider["base_url"]
        api_key = provider["api_key"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if provider["provider"] == "openrouter":
            headers["HTTP-Referer"] = "https://agroplan-colombia.dev"
            headers["X-Title"] = "AgroPlan Colombia"

        payload = self._build_chat_request(
            provider, model, system_prompt, user_content, response_format
        )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

                if response.status_code == 429:
                    return {"error": "rate_limited", "status": 429}

                if response.status_code >= 500:
                    return {"error": f"server_error_{response.status_code}", "status": response.status_code}

                response.raise_for_status()
                data = response.json()

                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})

                return {
                    "content": content,
                    "tokens_in": usage.get("prompt_tokens"),
                    "tokens_out": usage.get("completion_tokens"),
                    "model": model,
                    "provider": provider["provider"],
                }

        except httpx.TimeoutException:
            return {"error": "timeout"}
        except httpx.HTTPError as e:
            return {"error": f"http_error: {e}"}
        except Exception as e:
            return {"error": f"unexpected: {e}"}

    def _try_repair_json(self, content: str) -> Optional[Dict]:
        """Attempt one round of JSON repair."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                start = content.index("```json") + 7
                end = content.index("```", start)
                try:
                    return json.loads(content[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass
            if "```" in content:
                start = content.index("```") + 3
                end = content.index("```", start)
                try:
                    return json.loads(content[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass
            return None

    def generate_explanation(
        self,
        prediction_data: Dict[str, Any],
        municipality_name: str,
    ) -> Dict[str, Any]:
        """Generate an LLM explanation for prediction results.

        Returns a dict with:
        - explanation: str or None
        - llm_status: "success" or "llm_unavailable"
        - provider: str or None
        - model: str or None
        - tokens_in: int or None
        - tokens_out: int or None
        - latency_ms: int or None
        - error: str or None
        """
        start = time.time()

        system_prompt = (
            "You are an agronomic assistant for Colombia. Given crop prediction data, "
            "write a concise, practical explanation in Spanish for a farmer. "
            "Focus on: why the crop is or isn't suitable, key climate factors, and "
            "one actionable recommendation. Do not invent data. Use only the provided "
            "information. Keep it under 200 words."
        )

        user_content = json.dumps(
            {
                "municipality": municipality_name,
                "predictions": prediction_data,
            },
            ensure_ascii=False,
            default=str,
        )

        providers = self._get_providers()

        if not providers:
            latency_ms = int((time.time() - start) * 1000)
            return {
                "explanation": None,
                "llm_status": "llm_unavailable",
                "provider": None,
                "model": None,
                "tokens_in": None,
                "tokens_out": None,
                "latency_ms": latency_ms,
                "error": "No LLM providers configured",
            }

        for provider in providers:
            for model in provider["models"]:
                result = self._call_provider(
                    provider, model, system_prompt, user_content
                )

                if result is None:
                    continue

                if "error" in result:
                    # Try next model/provider
                    print(f"[llm] {provider['provider']}/{model} failed: {result['error']}")
                    continue

                latency_ms = int((time.time() - start) * 1000)
                explanation = result.get("content", "").strip()

                if explanation:
                    return {
                        "explanation": explanation,
                        "llm_status": "success",
                        "provider": result.get("provider"),
                        "model": result.get("model"),
                        "tokens_in": result.get("tokens_in"),
                        "tokens_out": result.get("tokens_out"),
                        "latency_ms": latency_ms,
                        "error": None,
                    }

        latency_ms = int((time.time() - start) * 1000)
        return {
            "explanation": None,
            "llm_status": "llm_unavailable",
            "provider": None,
            "model": None,
            "tokens_in": None,
            "tokens_out": None,
            "latency_ms": latency_ms,
            "error": "All providers failed",
        }

    def is_configured(self) -> bool:
        """Check if at least one LLM provider is configured."""
        return len(self._get_providers()) > 0


# Singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
