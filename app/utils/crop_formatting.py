"""Utilities for formatting crop data into farmer-friendly text."""

import re
from typing import Optional


def format_duration_days(days: Optional[int]) -> str:
    """Convert a number of days into an easy-to-read duration.

    Examples:
        60  -> "2 meses"
        45  -> "1 mes y 2 semanas"
        365 -> "1 año"
        730 -> "2 años"
        365*5 + 180 -> "5 años y 6 meses"
        30  -> "1 mes"
        14  -> "2 semanas"
        7   -> "1 semana"
        1   -> "1 día"
    """
    if days is None or days <= 0:
        return ""

    years = days // 365
    remaining = days % 365

    months = remaining // 30
    remaining = remaining % 30

    weeks = remaining // 7
    remaining_days = remaining % 7

    parts = []
    if years == 1:
        parts.append("1 año")
    elif years > 1:
        parts.append(f"{years} años")

    if months == 1:
        parts.append("1 mes")
    elif months > 1:
        parts.append(f"{months} meses")

    if weeks == 1:
        parts.append("1 semana")
    elif weeks > 1:
        parts.append(f"{weeks} semanas")

    if remaining_days == 1:
        parts.append("1 día")
    elif remaining_days > 1:
        parts.append(f"{remaining_days} días")

    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return " y ".join([", ".join(parts[:-1]), parts[-1]])


def _acidity_descriptor(match: re.Match) -> str:
    """Map a pH/acidity range to a simple descriptor."""
    try:
        low = float(match.group(1).replace(",", "."))
        high = float(match.group(2).replace(",", "."))
    except (ValueError, AttributeError):
        return match.group(0)

    # pH scale: <7 acidic, 7 neutral, >7 alkaline
    if low >= 6.5 and high <= 7.5:
        return "acidez neutra"
    if low >= 7.0:
        return "acidez neutra a fuerte"
    if high <= 5.5:
        return "acidez suave"
    if low >= 5.0 and high <= 6.5:
        return "acidez suave"
    if low < 7.0 <= high:
        return "acidez de suave a neutra"
    return "acidez de suave a fuerte"


def simplify_soil_terms(text: str) -> str:
    """Replace technical soil terminology with farmer-friendly words.

    This is intentionally conservative: only well-known technical terms are
    replaced so the meaning stays clear.
    """
    if not text:
        return text

    result = text

    # Replace specific compound terms first, before the generic ones.
    specific_replacements = {
        r"\bfranco a franco arcilloso\b": "tierra mezclada o de arcilla",
        r"\bfranco arcilloso\b": "tierra de arcilla",
        r"\bsuelo franco arenoso\b": "tierra mezclada de arena y barro",
        r"\bfranco arenoso\b": "tierra mezclada de arena y barro",
        r"\bsuelo franco\b": "tierra mezclada",
        r"\bfranco\b": "tierra mezclada",
        r"\bbien drenado\b": "que deja salir bien el agua",
        r"\bbuen drenaje\b": "buen drenaje del agua",
        r"\bprofundo\b": "profunda",
        r"\bmsnm\b": "sobre el nivel del mar",
        r"\bmetros sobre el nivel del mar\b": "sobre el nivel del mar",
        r"\bbuena materia orgánica\b": "buen abono natural o tierra negra",
        r"\bbuena materia organica\b": "buen abono natural o tierra negra",
        r"\bmateria orgánica\b": "abono natural o tierra negra",
        r"\bmateria organica\b": "abono natural o tierra negra",
    }

    for pattern, replacement in specific_replacements.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Replace pH ranges like "pH 4.5 - 7.0" or "pH 5.5-7.0" with a descriptor.
    result = re.sub(
        r"pH\s*([0-9]+[.,]?[0-9]*)\s*-\s*([0-9]+[.,]?[0-9]*)",
        _acidity_descriptor,
        result,
        flags=re.IGNORECASE,
    )

    # Replace remaining "pH" mentions not followed by a number range.
    result = re.sub(r"\bpH\b", "acidez", result, flags=re.IGNORECASE)

    # Clean up double spaces or trailing punctuation artifacts.
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _temperature_descriptor(match: re.Match) -> str:
    """Map a Celsius temperature range to a farmer-friendly descriptor."""
    try:
        low = float(match.group(1).replace(",", "."))
        high = float(match.group(2).replace(",", "."))
    except (ValueError, AttributeError):
        return match.group(0)

    if high <= 15:
        return "frío"
    if low >= 25:
        return "calor"
    if low >= 15 and high <= 25:
        return "templado"
    if low < 15 and high >= 25:
        return "frío a caluroso"
    if low < 15 <= high:
        return "frío a templado"
    if low < 25 <= high:
        return "templado a caluroso"
    return "templado"


def simplify_temperature(text: str) -> str:
    """Replace temperature ranges with plain-language descriptors."""
    if not text:
        return text

    result = re.sub(
        r"([0-9]+[.,]?[0-9]*)\s*-\s*([0-9]+[.,]?[0-9]*)\s*C",
        _temperature_descriptor,
        text,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _humidity_descriptor(match: re.Match) -> str:
    """Map a humidity percentage range to a simple phrase."""
    try:
        low = float(match.group(1).replace(",", "."))
        high = float(match.group(2).replace(",", "."))
    except (ValueError, AttributeError):
        return match.group(0)

    avg = (low + high) / 2
    if avg < 50:
        return "ambiente seco"
    if avg < 70:
        return "ambiente moderadamente húmedo"
    return "ambiente húmedo"


def simplify_humidity(text: str) -> str:
    """Replace humidity percentage ranges with plain-language phrases."""
    if not text:
        return text

    result = re.sub(
        r"([0-9]+[.,]?[0-9]*)\s*-\s*([0-9]+[.,]?[0-9]*)\s*%",
        _humidity_descriptor,
        text,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _precipitation_descriptor(match: re.Match) -> str:
    """Map a precipitation range to a simple rainfall phrase."""
    try:
        low = float(match.group(1).replace(".", "").replace(",", "."))
        high = float(match.group(2).replace(".", "").replace(",", "."))
    except (ValueError, AttributeError):
        return match.group(0)

    avg = (low + high) / 2
    if avg < 600:
        return "poca lluvia"
    if avg < 1500:
        return "lluvia moderada"
    if avg < 2500:
        return "buena lluvia"
    return "mucha lluvia"


def simplify_precipitation(text: str) -> str:
    """Replace precipitation ranges with plain-language rainfall phrases."""
    if not text:
        return text

    # Normalize first: replace "/ano" and "/ciclo" with nothing special,
    # the descriptor already implies it.
    result = re.sub(r"/\s*ano\b", "", text, flags=re.IGNORECASE)
    result = re.sub(r"/\s*ciclo\b", "", result, flags=re.IGNORECASE)

    result = re.sub(
        r"([0-9]+[.,]?[0-9]*)\s*-\s*([0-9]+[.,]?[0-9]*)\s*mm",
        _precipitation_descriptor,
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _altitude_descriptor(match: re.Match) -> str:
    """Map an altitude range to a simple thermal-floor phrase."""
    try:
        low = float(match.group(1).replace(".", "").replace(",", "."))
        high = float(match.group(2).replace(".", "").replace(",", "."))
    except (ValueError, AttributeError):
        return match.group(0)

    if high <= 1000:
        return "zona cálida"
    if low >= 2000:
        return "zona fría"
    if low >= 1000 and high <= 2000:
        return "zona templada"
    if low < 1000 <= high:
        return "zona cálida a templada"
    if low < 2000 <= high:
        return "zona templada a fría"
    return "zona templada"


def simplify_altitude(text: str) -> str:
    """Replace altitude ranges with plain-language thermal-floor phrases."""
    if not text:
        return text

    result = re.sub(
        r"([0-9]+[.,]?[0-9]*)\s*-\s*([0-9]+[.,]?[0-9]*)\s*(msnm|metros?)",
        _altitude_descriptor,
        text,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\s+", " ", result).strip()
    return result
