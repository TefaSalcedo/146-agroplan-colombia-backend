"""LLM service for generating explanations and validating prediction results.

Supports OpenRouter and Groq with round-robin model selection across providers.
When all LLM providers fail, returns None and the caller returns a 200
with data intact and explanation fields set to null.
"""

import json
import threading
import time
from typing import Optional, Dict, Any, List

import httpx

from app.config import get_settings

settings = get_settings()

PROMPT_SCHEMA_VERSION = "1.0"


class LLMService:
    """LLM service with round-robin provider/model selection."""

    def __init__(self):
        self.timeout = settings.llm_timeout_seconds
        self._round_robin_index = 0
        self._round_robin_lock = threading.Lock()

    def _openrouter_config(self) -> Optional[Dict[str, Any]]:
        models = settings.openrouter_models_list
        if not settings.openrouter_api_key or not models:
            return None
        return {
            "provider": "openrouter",
            "api_key": settings.openrouter_api_key,
            "models": models,
            "base_url": "https://openrouter.ai/api/v1",
        }

    def _groq_config(self) -> Optional[Dict[str, Any]]:
        models = settings.groq_models_list
        if not settings.groq_api_key or not models:
            return None
        return {
            "provider": "groq",
            "api_key": settings.groq_api_key,
            "models": models,
            "base_url": "https://api.groq.com/openai/v1",
        }

    def _get_model_pool(self) -> List[Dict[str, Any]]:
        """Return a flat list of all configured (provider, model) entries."""
        pool: List[Dict[str, Any]] = []

        openrouter = self._openrouter_config()
        groq = self._groq_config()

        primary = settings.llm_provider.lower()

        # Round-robin interleaves providers starting from the preferred one.
        if primary == "openrouter":
            first, second = openrouter, groq
        else:
            first, second = groq, openrouter

        if first:
            for model in first["models"]:
                pool.append({
                    "provider": first["provider"],
                    "api_key": first["api_key"],
                    "base_url": first["base_url"],
                    "model": model,
                })

        if second:
            for model in second["models"]:
                pool.append({
                    "provider": second["provider"],
                    "api_key": second["api_key"],
                    "base_url": second["base_url"],
                    "model": model,
                })

        return pool

    def _select_next_model(self, pool: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Pick the next model in the round-robin sequence."""
        if not pool:
            return None
        with self._round_robin_lock:
            idx = self._round_robin_index % len(pool)
            self._round_robin_index = (self._round_robin_index + 1) % len(pool)
            return pool[idx]

    def _build_system_prompt(self, base_prompt: str, response_format: Optional[Dict] = None) -> str:
        """Append Spanish-language instruction unless the response must be structured JSON."""
        if response_format is not None:
            return base_prompt
        return (
            f"{base_prompt}\n\n"
            "Responde siempre en español."
        )

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
            {"role": "system", "content": self._build_system_prompt(system_prompt, response_format)},
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
            "write a concise, practical explanation for a farmer. "
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

        pool = self._get_model_pool()

        if not pool:
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

        # Round-robin: start from the next model in the pool.
        first_selected = self._select_next_model(pool)
        if first_selected is None:
            return {
                "explanation": None,
                "llm_status": "llm_unavailable",
                "provider": None,
                "model": None,
                "tokens_in": None,
                "tokens_out": None,
                "latency_ms": int((time.time() - start) * 1000),
                "error": "No LLM models available",
            }

        # Build a circular iterator starting from the selected model so that
        # failures continue with the next one in round-robin order.
        start_idx = pool.index(first_selected)
        ordered_pool = pool[start_idx:] + pool[:start_idx]

        for entry in ordered_pool:
            provider = {
                "provider": entry["provider"],
                "api_key": entry["api_key"],
                "base_url": entry["base_url"],
            }
            model = entry["model"]

            result = self._call_provider(
                provider, model, system_prompt, user_content
            )

            if result is None:
                continue

            if "error" in result:
                print(f"[llm] {entry['provider']}/{model} failed: {result['error']}")
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
        return len(self._get_model_pool()) > 0


# Singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
