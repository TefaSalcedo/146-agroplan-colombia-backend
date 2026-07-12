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

from sqlalchemy.orm import Session

from app.config import get_settings
from app.logger import get_logger

settings = get_settings()
logger = get_logger("app.services.llm_service")

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
        """Return a flat list of all configured (provider, model) entries.

        Providers are interleaved so that a failure in one provider immediately
        falls back to the other provider, rather than exhausting every model
        from the preferred provider first.
        """
        pool: List[Dict[str, Any]] = []

        openrouter = self._openrouter_config()
        groq = self._groq_config()

        primary = settings.llm_provider.lower()
        preferred = openrouter if primary == "openrouter" else groq
        fallback = groq if primary == "openrouter" else openrouter

        preferred_models = preferred["models"] if preferred else []
        fallback_models = fallback["models"] if fallback else []
        max_len = max(len(preferred_models), len(fallback_models))

        for i in range(max_len):
            if i < len(preferred_models):
                pool.append({
                    "provider": preferred["provider"],
                    "api_key": preferred["api_key"],
                    "base_url": preferred["base_url"],
                    "model": preferred_models[i],
                })
            if i < len(fallback_models):
                pool.append({
                    "provider": fallback["provider"],
                    "api_key": fallback["api_key"],
                    "base_url": fallback["base_url"],
                    "model": fallback_models[i],
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
        logger.debug("[generate_explanation] LLM pool has %s entries", len(pool))

        if not pool:
            latency_ms = int((time.time() - start) * 1000)
            logger.warning("[generate_explanation] No LLM providers configured")
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
            logger.warning("[generate_explanation] No LLM models available")
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

        logger.info("[generate_explanation] Starting round-robin LLM calls (first=%s/%s)", first_selected["provider"], first_selected["model"])

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
            logger.info("[generate_explanation] Calling LLM provider=%s model=%s", provider["provider"], model)

            result = self._call_provider(
                provider, model, system_prompt, user_content
            )

            if result is None:
                logger.warning("[generate_explanation] Provider %s/%s returned None", provider["provider"], model)
                continue

            if "error" in result:
                logger.warning("[generate_explanation] Provider %s/%s failed: %s", provider["provider"], model, result["error"])
                continue

            latency_ms = int((time.time() - start) * 1000)
            explanation = result.get("content", "").strip()
            logger.info("[generate_explanation] Provider %s/%s succeeded (latency_ms=%s, tokens_in=%s, tokens_out=%s)", provider["provider"], model, latency_ms, result.get("tokens_in"), result.get("tokens_out"))

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
        logger.warning("[generate_explanation] All LLM providers failed after round-robin (latency_ms=%s)", latency_ms)
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

    def generate_crop_recommendation(
        self,
        crop: Dict[str, Any],
        municipality: Dict[str, Any],
        climate_summary: Dict[str, Any],
        soil_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a short, practical recommendation for a crop in a municipality.

        Returns the same audit fields as generate_explanation plus a farmer-facing
        recommendation text.
        """
        start = time.time()

        system_prompt = (
            "Eres un asistente agronomo para campesinos de Colombia. "
            "Responde SIEMPRE en espanol, en un solo parrafo corto (maximo 120 palabras). "
            "Usa lenguaje sencillo y util para el dia a dia del agricultor. "
            "Da UNA recomendacion clara: si conviene sembrar este cultivo en este municipio, "
            "en que meses plantar, cuidados basicos de suelo/riego, y advertencias si el clima "
            "o suelo no son adecuados. No inventes datos; usa solo la informacion proporcionada."
        )

        user_content = json.dumps(
            {
                "cultivo": crop,
                "municipio": municipality,
                "clima": climate_summary,
                "suelo": soil_summary,
            },
            ensure_ascii=False,
            default=str,
        )

        pool = self._get_model_pool()
        logger.debug("[generate_crop_recommendation] LLM pool has %s entries", len(pool))

        if not pool:
            latency_ms = int((time.time() - start) * 1000)
            logger.warning("[generate_crop_recommendation] No LLM providers configured")
            return {
                "text": "",
                "status": "llm_unavailable",
                "provider": None,
                "model": None,
                "tokens_in": None,
                "tokens_out": None,
                "latency_ms": latency_ms,
                "error": "No LLM providers configured",
            }

        first_selected = self._select_next_model(pool)
        if first_selected is None:
            return {
                "text": "",
                "status": "llm_unavailable",
                "provider": None,
                "model": None,
                "tokens_in": None,
                "tokens_out": None,
                "latency_ms": int((time.time() - start) * 1000),
                "error": "No LLM models available",
            }

        logger.info("[generate_crop_recommendation] Starting round-robin LLM calls (first=%s/%s)", first_selected["provider"], first_selected["model"])

        start_idx = pool.index(first_selected)
        ordered_pool = pool[start_idx:] + pool[:start_idx]

        for entry in ordered_pool:
            provider = {
                "provider": entry["provider"],
                "api_key": entry["api_key"],
                "base_url": entry["base_url"],
            }
            model = entry["model"]
            logger.info("[generate_crop_recommendation] Calling LLM provider=%s model=%s", provider["provider"], model)

            result = self._call_provider(provider, model, system_prompt, user_content)

            if result is None:
                logger.warning("[generate_crop_recommendation] Provider %s/%s returned None", provider["provider"], model)
                continue

            if "error" in result:
                logger.warning("[generate_crop_recommendation] Provider %s/%s failed: %s", provider["provider"], model, result["error"])
                continue

            latency_ms = int((time.time() - start) * 1000)
            text = result.get("content", "").strip()
            logger.info("[generate_crop_recommendation] Provider %s/%s succeeded (latency_ms=%s, tokens_in=%s, tokens_out=%s)", provider["provider"], model, latency_ms, result.get("tokens_in"), result.get("tokens_out"))

            return {
                "text": text,
                "status": "success",
                "provider": result.get("provider"),
                "model": result.get("model"),
                "tokens_in": result.get("tokens_in"),
                "tokens_out": result.get("tokens_out"),
                "latency_ms": latency_ms,
                "error": None,
            }

        latency_ms = int((time.time() - start) * 1000)
        logger.warning("[generate_crop_recommendation] All LLM providers failed after round-robin (latency_ms=%s)", latency_ms)
        return {
            "text": "",
            "status": "llm_unavailable",
            "provider": None,
            "model": None,
            "tokens_in": None,
            "tokens_out": None,
            "latency_ms": latency_ms,
            "error": "All providers failed",
        }

    def generate_national_crop_guide(
        self,
        crop_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a structured national farmer guide for a crop.

        The guide is written in simple Spanish for Colombian farmers and covers
        planting, harvest, techniques and seasons. The response is a JSON object
        with a summary and a list of sections.
        """
        start = time.time()

        system_prompt = (
            "Eres un agrónomo colombiano con mucha experiencia en extensión rural. "
            "Tu trabajo es escribir una guía práctica y fácil de leer para campesinos "
            "de Colombia que quieren sembrar un cultivo.\n\n"
            "REGLAS IMPORTANTES:\n"
            "1. Usa un lenguaje sencillo, cercano y sin tecnicismos difíciles.\n"
            "2. Escribe en español de Colombia.\n"
            "3. No inventes datos. Usa únicamente la información del cultivo proporcionada.\n"
            "4. La guía debe tener un resumen corto y varias secciones con título y contenido.\n\n"
            "ESTRUCTURA OBLIGATORIA DEL JSON:\n"
            "{\n"
            '  "summary": "resumen amigable de 2 o 3 frases",\n'
            '  "sections": [\n'
            '    {"title": "¿Cuándo sembrar?", "content": "..."},\n'
            '    {"title": "¿Cuánto tarda la cosecha?", "content": "..."},\n'
            '    {"title": "Preparación del terreno", "content": "..."},\n'
            '    {"title": "Riego y cuidados básicos", "content": "..."},\n'
            '    {"title": "Plagas y enfermedades comunes", "content": "..."},\n'
            '    {"title": "Cosecha y post-cosecha", "content": "..."},\n'
            '    {"title": "Advertencias importantes", "content": "..."}\n'
            "  ]\n"
            "}\n\n"
            "Las secciones deben ser concretas, útiles y fáciles de entender para una "
            "persona que trabaja la tierra todos los días."
        )

        user_content = json.dumps(
            {"crop": crop_data},
            ensure_ascii=False,
            default=str,
        )

        pool = self._get_model_pool()
        if not pool:
            return {
                "summary": "",
                "sections": [],
                "status": "llm_unavailable",
                "provider": None,
                "model": None,
                "tokens_in": None,
                "tokens_out": None,
                "latency_ms": int((time.time() - start) * 1000),
                "error": "No LLM providers configured",
            }

        first_selected = self._select_next_model(pool)
        start_idx = pool.index(first_selected)
        ordered_pool = pool[start_idx:] + pool[:start_idx]

        for entry in ordered_pool:
            provider = {
                "provider": entry["provider"],
                "api_key": entry["api_key"],
                "base_url": entry["base_url"],
            }
            model = entry["model"]
            logger.info("[generate_national_crop_guide] Calling LLM provider=%s model=%s", provider["provider"], model)

            result = self._call_provider(
                provider,
                model,
                system_prompt,
                user_content,
                response_format={"type": "json_object"},
            )

            if result is None or "error" in result:
                logger.warning(
                    "[generate_national_crop_guide] Provider %s/%s failed: %s",
                    provider["provider"],
                    model,
                    result.get("error") if result else "None",
                )
                continue

            latency_ms = int((time.time() - start) * 1000)
            raw_content = result.get("content", "").strip()
            if not raw_content:
                logger.warning(
                    "[generate_national_crop_guide] Provider %s/%s returned empty content",
                    provider["provider"],
                    model,
                )
                continue

            try:
                parsed = self._try_repair_json(raw_content)
            except Exception as exc:
                logger.warning(
                    "[generate_national_crop_guide] Provider %s/%s returned unparseable content: %s",
                    provider["provider"],
                    model,
                    exc,
                )
                continue

            if not parsed or not isinstance(parsed, dict):
                logger.warning(
                    "[generate_national_crop_guide] Provider %s/%s returned non-JSON content",
                    provider["provider"],
                    model,
                )
                continue

            summary = parsed.get("summary", "").strip()
            sections = parsed.get("sections", [])
            if not isinstance(sections, list):
                sections = []

            normalized_sections = []
            for sec in sections:
                if isinstance(sec, dict) and sec.get("title") and sec.get("content"):
                    normalized_sections.append({
                        "title": str(sec["title"]).strip(),
                        "content": str(sec["content"]).strip(),
                    })

            logger.info(
                "[generate_national_crop_guide] Provider %s/%s succeeded (latency_ms=%s, tokens_in=%s, tokens_out=%s)",
                provider["provider"],
                model,
                latency_ms,
                result.get("tokens_in"),
                result.get("tokens_out"),
            )

            return {
                "summary": summary,
                "sections": normalized_sections,
                "status": "success",
                "provider": result.get("provider"),
                "model": result.get("model"),
                "tokens_in": result.get("tokens_in"),
                "tokens_out": result.get("tokens_out"),
                "latency_ms": latency_ms,
                "error": None,
            }

        return {
            "summary": "",
            "sections": [],
            "status": "llm_unavailable",
            "provider": None,
            "model": None,
            "tokens_in": None,
            "tokens_out": None,
            "latency_ms": int((time.time() - start) * 1000),
            "error": "All providers failed",
        }

    def generate_municipality_ai_guide(
        self,
        municipality_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate AI-driven recommendations for a municipality.

        Returns structured insights: alternative crops, farming systems and soil/
        fertilizer advice written in plain Spanish for Colombian farmers.
        """
        start = time.time()

        system_prompt = (
            "Eres un agrónomo colombiano que asesora a campesinos de todo el país. "
            "Tu trabajo es dar recomendaciones PRÁCTICAS y FÁCILES DE ENTENDER para un "
            "municipio específico.\n\n"
            "REGLAS DE LENGUAJE:\n"
            "1. Usa palabras comunes del campo. NO uses términos técnicos como "
            "'suelo franco arenoso', 'textura franco', 'pH', 'topografía', 'latitud'.\n"
            "2. Si necesitas hablar del suelo, di cosas como 'tierra suelta', 'tierra "
            "pesada', 'tierra que drena bien', 'tierra negra y fértil', etc.\n"
            "3. Si hablas de altura, di 'esta zona es de montaña baja', 'zona fría', "
            "'zona templada', 'zona cálida', según corresponda.\n"
            "4. La temperatura exprésala como 'hace frío', 'hace calor', 'es templado'. "
            "No uses grados Celsius a menos que sea necesario.\n"
            "5. Todo en español de Colombia.\n\n"
            "REGLAS DE CONTENIDO:\n"
            "1. En 'alternative_crops' sugiere cultivos que NO estén en la lista del "
            "municipio. Como base de inspiración usa la lista 'alternative_crops_from_ecocrop_fao' "
            "que ya filtré por clima, altura y lluvia del municipio. Puedes incluir cultivos de "
            "esa lista y complementar con tu conocimiento agronómico cuando sea necesario. "
            "Explica por qué cada cultivo puede funcionar allí.\n"
            "2. En 'farming_systems' recomienda cualquier sistema de producción que tenga "
            "sentido para el municipio: viveros, semilleros, hidroponía, aboneras, "
            "cultivo en eras o camellones, zanjas, riego por goteo, riego por gravedad, "
            "aspersión, invernadero o cubierta de plástico, asociación de cultivos, "
            "rotación de cultivos, barreras vivas, sistema de drenaje, protección contra "
            "heladas, etc. Elige solo los que realmente apliquen y explica por qué. "
            "Si alguno no es conveniente, indícalo con suitable='no'.\n"
            "3. En 'soil_and_fertilizer' da consejos sencillos sobre abonos orgánicos, "
            "cómo mejorar la tierra, cuándo fertilizar, etc.\n\n"
            "ESTRUCTURA OBLIGATORIA DEL JSON:\n"
            "{\n"
            '  "summary": "resumen de 2 o 3 frases",\n'
            '  "alternative_crops": [\n'
            '    {"crop_name": "nombre común", "why": "por qué sirve aquí", "confidence": "high|medium|low"}\n'
            '  ],\n'
            '  "farming_systems": [\n'
            '    {"title": "Vivero", "recommendation": "...", "suitable": "yes|partial|no"}\n'
            '  ],\n'
            '  "soil_and_fertilizer": [\n'
            '    {"title": "título", "content": "..."}\n'
            '  ]\n'
            "}\n\n"
            "Las recomendaciones deben ser concretas, seguras y útiles para una persona que "
            "trabaja directamente la tierra."
        )

        user_content = json.dumps(
            {"municipality": municipality_data},
            ensure_ascii=False,
            default=str,
        )

        pool = self._get_model_pool()
        if not pool:
            return {
                "summary": "",
                "alternative_crops": [],
                "farming_systems": [],
                "soil_and_fertilizer": [],
                "status": "llm_unavailable",
                "provider": None,
                "model": None,
                "tokens_in": None,
                "tokens_out": None,
                "latency_ms": int((time.time() - start) * 1000),
                "error": "No LLM providers configured",
            }

        first_selected = self._select_next_model(pool)
        start_idx = pool.index(first_selected)
        ordered_pool = pool[start_idx:] + pool[:start_idx]

        for entry in ordered_pool:
            provider = {
                "provider": entry["provider"],
                "api_key": entry["api_key"],
                "base_url": entry["base_url"],
            }
            model = entry["model"]
            logger.info(
                "[generate_municipality_ai_guide] Calling LLM provider=%s model=%s",
                provider["provider"],
                model,
            )

            result = self._call_provider(
                provider,
                model,
                system_prompt,
                user_content,
                response_format={"type": "json_object"},
            )

            if result is None or "error" in result:
                logger.warning(
                    "[generate_municipality_ai_guide] Provider %s/%s failed: %s",
                    provider["provider"],
                    model,
                    result.get("error") if result else "None",
                )
                continue

            latency_ms = int((time.time() - start) * 1000)
            raw_content = result.get("content", "").strip()
            if not raw_content:
                logger.warning(
                    "[generate_municipality_ai_guide] Provider %s/%s returned empty content",
                    provider["provider"],
                    model,
                )
                continue

            try:
                parsed = self._try_repair_json(raw_content)
            except Exception as exc:
                logger.warning(
                    "[generate_municipality_ai_guide] Provider %s/%s returned unparseable content: %s",
                    provider["provider"],
                    model,
                    exc,
                )
                continue

            if not parsed or not isinstance(parsed, dict):
                logger.warning(
                    "[generate_municipality_ai_guide] Provider %s/%s returned non-JSON content",
                    provider["provider"],
                    model,
                )
                continue

            logger.info(
                "[generate_municipality_ai_guide] Provider %s/%s succeeded (latency_ms=%s, tokens_in=%s, tokens_out=%s)",
                provider["provider"],
                model,
                latency_ms,
                result.get("tokens_in"),
                result.get("tokens_out"),
            )

            return {
                "summary": parsed.get("summary", "").strip(),
                "alternative_crops": self._normalize_list(parsed.get("alternative_crops", [])),
                "farming_systems": self._normalize_list(parsed.get("farming_systems", [])),
                "soil_and_fertilizer": self._normalize_list(parsed.get("soil_and_fertilizer", [])),
                "status": "success",
                "provider": result.get("provider"),
                "model": result.get("model"),
                "tokens_in": result.get("tokens_in"),
                "tokens_out": result.get("tokens_out"),
                "latency_ms": latency_ms,
                "error": None,
            }

        return {
            "summary": "",
            "alternative_crops": [],
            "farming_systems": [],
            "soil_and_fertilizer": [],
            "status": "llm_unavailable",
            "provider": None,
            "model": None,
            "tokens_in": None,
            "tokens_out": None,
            "latency_ms": int((time.time() - start) * 1000),
            "error": "All providers failed",
        }

    @staticmethod
    def _normalize_list(value: Any) -> List[Dict[str, Any]]:
        """Return a clean list of dicts from an LLM JSON field."""
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            if isinstance(item, dict):
                cleaned.append({str(k): str(v).strip() for k, v in item.items()})
        return cleaned

    def is_configured(self) -> bool:
        """Check if at least one LLM provider is configured."""
        return len(self._get_model_pool()) > 0


# Singleton
_llm_service: Optional[LLMService] = None


def log_llm_generation(
    db: Session,
    *,
    provider: Optional[str],
    model: Optional[str],
    prompt_schema_version: str,
    context_summary: str,
    response_json: Optional[Dict[str, Any]],
    tokens_in: Optional[int],
    tokens_out: Optional[int],
    latency_ms: Optional[int],
    status: str,
    error_message: Optional[str] = None,
    prediction_run_id: Optional[int] = None,
) -> Optional[int]:
    """Persist an LLM call to the database for auditing and cost tracking.

    Returns the generated row id, or None if the insert fails.
    """
    try:
        from app.models import LLMGeneration

        generation = LLMGeneration(
            provider=provider or "unknown",
            model=model or "unknown",
            prompt_schema_version=prompt_schema_version,
            context_summary=context_summary,
            response_json=response_json,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
            prediction_run_id=prediction_run_id,
        )
        db.add(generation)
        db.commit()
        db.refresh(generation)
        logger.info("[log_llm_generation] Saved LLM generation id=%s provider=%s model=%s", generation.id, provider, model)
        return generation.id
    except Exception as e:
        logger.error("[log_llm_generation] Failed to persist LLM generation: %s", e)
        db.rollback()
        return None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
