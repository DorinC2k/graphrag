"""Utilities to add a medication by analyzing a package photo."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import pytesseract
import openai


@dataclass
class Medication:
    """Represents extracted medication information."""

    name: str
    expiration_date: str | None = None
    active_ingredient: str | None = None
    raw_text: str | None = None


def extract_text_from_image(image_path: str) -> str:
    """Return OCR text from a medication package image."""
    image = Image.open(Path(image_path))
    return pytesseract.image_to_string(image)


def _parse_expiration_date(text: str) -> str | None:
    patterns = [
        r"EXP(?:IRATION)?\s*DATE[:\s]*([0-9]{2}/[0-9]{2}/[0-9]{4})",
        r"EXP(?:IRATION)?\s*DATE[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2})",
        r"Expires[:\s]*([0-9]{2}/[0-9]{2}/[0-9]{4})",
        r"Expires[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def parse_medication_details(text: str) -> Medication:
    """Parse text to extract medication information."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    name = lines[0] if lines else ""
    expiration = _parse_expiration_date(text)
    return Medication(name=name, expiration_date=expiration, raw_text=text)


def derive_active_ingredient(name: str) -> str:
    """Use OpenAI to derive the active ingredient for the medication name."""
    prompt = f"What is the active ingredient in {name}?"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def add_medication(image_path: str) -> Medication:
    """Return medication object populated from image and OpenAI."""
    text = extract_text_from_image(image_path)
    med = parse_medication_details(text)
    try:
        med.active_ingredient = derive_active_ingredient(med.name)
    except Exception:
        med.active_ingredient = None
    return med
