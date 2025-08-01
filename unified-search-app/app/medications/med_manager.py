# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utilities to add medications by extracting data from package photos."""

import re
from dataclasses import dataclass

import pytesseract
from PIL import Image


@dataclass
class Medication:
    """Simple medication data model."""

    name: str
    expiration_date: str
    active_ingredient: str | None = None


def extract_text_from_image(image_path: str) -> str:
    """Return OCR text extracted from an image."""
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


def parse_expiration_date(text: str) -> str | None:
    """Return expiration date parsed from OCR text if present."""
    pattern = r"(?i)(?:exp(?:iration)?\s*date|expires?|exp)[:\s]*([0-9\-/]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def derive_active_ingredient(med_name: str) -> str:
    """Query OpenAI to derive the active ingredient for a medication name."""
    import openai

    client = openai.OpenAI()
    prompt = (
        "What is the active ingredient of the medication '"
        + med_name
        + "'? Provide only the ingredient name."
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
    )
    return response.choices[0].message.content.strip()


def add_medication(image_path: str) -> Medication:
    """Extract medication info from an image and enrich via OpenAI."""
    text = extract_text_from_image(image_path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    name = lines[0] if lines else "Unknown"
    expiration = parse_expiration_date(text) or ""
    active_ingredient = derive_active_ingredient(name)
    return Medication(name=name, expiration_date=expiration, active_ingredient=active_ingredient)
