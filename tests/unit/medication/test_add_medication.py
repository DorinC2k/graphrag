# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "unified-search-app"))

from app.medications import (
    add_medication,
    med_manager,
    parse_expiration_date,
)


def test_parse_expiration_date():
    text = "Medicine XYZ EXP 2025-12-31"
    assert parse_expiration_date(text) == "2025-12-31"


def test_add_medication(monkeypatch):
    def fake_extract(path: str) -> str:
        return "MedA\nExp 2024-10-10"

    def fake_derive(name: str) -> str:
        return "acetaminophen"

    monkeypatch.setattr(med_manager, "extract_text_from_image", fake_extract)
    monkeypatch.setattr(med_manager, "derive_active_ingredient", fake_derive)

    med = add_medication("dummy.png")
    assert med.name == "MedA"
    assert med.expiration_date == "2024-10-10"
    assert med.active_ingredient == "acetaminophen"
