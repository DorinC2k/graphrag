from pathlib import Path
from unittest import mock

import openai

import sys
sys.path.append(str(Path(__file__).resolve().parents[3] / "unified-search-app"))

import importlib
add_med_module = importlib.import_module("app.medication.add_medication")
from app.medication.add_medication import (
    Medication,
    add_medication,
    derive_active_ingredient,
    parse_medication_details,
)


SAMPLE_TEXT = """Aspirin 100mg
EXP DATE: 2025-01-02"""


def test_parse_medication_details():
    med = parse_medication_details(SAMPLE_TEXT)
    assert med.name == "Aspirin 100mg"
    assert med.expiration_date == "2025-01-02"


def test_derive_active_ingredient(monkeypatch):
    class Resp:
        choices = [mock.Mock(message=mock.Mock(content="Acetylsalicylic Acid"))]

    monkeypatch.setattr(openai.ChatCompletion, "create", mock.Mock(return_value=Resp()))
    result = derive_active_ingredient("Aspirin")
    assert result == "Acetylsalicylic Acid"


def test_add_medication(monkeypatch):
    class Resp:
        choices = [mock.Mock(message=mock.Mock(content="Acetylsalicylic Acid"))]

    monkeypatch.setattr(openai.ChatCompletion, "create", mock.Mock(return_value=Resp()))
    monkeypatch.setattr(add_med_module, "extract_text_from_image", mock.Mock(return_value=SAMPLE_TEXT))
    med = add_medication("dummy.jpg")
    assert isinstance(med, Medication)
    assert med.name == "Aspirin 100mg"
    assert med.expiration_date == "2025-01-02"
    assert med.active_ingredient == "Acetylsalicylic Acid"
