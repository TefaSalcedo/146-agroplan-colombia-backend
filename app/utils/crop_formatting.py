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
        r"\bfranco a franco arcilloso\b": "tierra mezclada a tierra de arcilla",
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
