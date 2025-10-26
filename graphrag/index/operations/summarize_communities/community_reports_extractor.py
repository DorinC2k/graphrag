# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'CommunityReportsResult' and 'CommunityReportsExtractor' models."""

import json
import logging
import re
import traceback
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT

log = logging.getLogger(__name__)

# these tokens are used in the prompt
INPUT_TEXT_KEY = "input_text"
MAX_LENGTH_KEY = "max_report_length"


class FindingModel(BaseModel):
    """A model for the expected LLM response shape."""

    summary: str = Field(description="The summary of the finding.")
    explanation: str = Field(description="An explanation of the finding.")


class CommunityReportResponse(BaseModel):
    """A model for the expected LLM response shape."""

    title: str = Field(description="The title of the report.")
    summary: str = Field(description="A summary of the report.")
    findings: list[FindingModel] = Field(
        description="A list of findings in the report."
    )
    rating: float = Field(description="The rating of the report.")
    rating_explanation: str = Field(description="An explanation of the rating.")


@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    output: str
    structured_output: CommunityReportResponse | None


class CommunityReportsExtractor:
    """Community reports extractor class definition."""

    _model: ChatModel
    _extraction_prompt: str
    _output_formatter_prompt: str
    _on_error: ErrorHandlerFn
    _max_report_length: int

    def __init__(
        self,
        model_invoker: ChatModel,
        extraction_prompt: str | None = None,
        on_error: ErrorHandlerFn | None = None,
        max_report_length: int | None = None,
    ):
        """Init method definition."""
        self._model = model_invoker
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_report_length = max_report_length or 1500

    async def __call__(self, input_text: str):
        """Call method definition."""
        output = None
        try:
            prompt = self._format_prompt(
                self._extraction_prompt,
                {
                    INPUT_TEXT_KEY: input_text,
                    MAX_LENGTH_KEY: str(self._max_report_length),
                },
            )
            response = await self._model.achat(
                prompt,
                json=True,  # Leaving this as True to avoid creating new cache entries
                name="create_community_report",
                json_model=CommunityReportResponse,  # A model is required when using json mode
            )

            output = response.parsed_response
        except Exception as e:
            log.exception("error generating community report")
            self._on_error(e, traceback.format_exc(), None)

        text_output = self._get_text_output(output) if output else ""
        return CommunityReportsResult(
            structured_output=output,
            output=text_output,
        )

    def _format_prompt(self, prompt: str, values: dict[str, str]) -> str:
        """Safely substitute known placeholders in the prompt."""

        if not values:
            return prompt

        str_values = {key: str(value) for key, value in values.items()}

        pattern = re.compile(
            "|".join(
                re.escape("{" + key + "}")
                for key in str_values
            )
        )

        def replace(match: re.Match[str]) -> str:
            key = match.group(0)[1:-1]
            return str_values.get(key, match.group(0))

        return pattern.sub(replace, prompt)

    @staticmethod
    def _model_dump(model: Any) -> dict[str, Any]:
        """Return a dictionary representation for pydantic v1/v2 objects."""

        if model is None:
            return {}

        if isinstance(model, BaseModel):
            dump = getattr(model, "model_dump", None)
            if callable(dump):
                return dump()

            dump = getattr(model, "dict", None)
            if callable(dump):
                return dump()

        if isinstance(model, dict):
            return model

        return {}

    @staticmethod
    def _model_dump_json(model: Any, **kwargs: Any) -> str:
        """Return a JSON representation for pydantic v1/v2 objects."""

        if model is None:
            return json.dumps({}, **kwargs)

        if isinstance(model, BaseModel):
            dump_json = getattr(model, "model_dump_json", None)
            if callable(dump_json):
                return dump_json(**kwargs)

            dump_json = getattr(model, "json", None)
            if callable(dump_json):
                return dump_json(**kwargs)

        if isinstance(model, str):
            return model

        return json.dumps(CommunityReportsExtractor._model_dump(model), **kwargs)

    @staticmethod
    def _normalize_findings(raw_findings: Any) -> list[dict[str, str]]:
        """Normalize findings objects into dictionaries."""

        if not raw_findings:
            return []

        normalized: list[dict[str, str]] = []
        for finding in raw_findings:
            finding_dict = CommunityReportsExtractor._model_dump(finding)
            if not finding_dict:
                continue
            normalized.append(
                {
                    "summary": str(finding_dict.get("summary", "")),
                    "explanation": str(finding_dict.get("explanation", "")),
                }
            )

        return normalized

    def _get_text_output(self, report: CommunityReportResponse) -> str:
        report_data = self._model_dump(report)
        findings = self._normalize_findings(report_data.get("findings"))
        report_sections = "\n\n".join(
            f"## {finding['summary']}\n\n{finding['explanation']}"
            for finding in findings
        )

        title = str(report_data.get("title", ""))
        summary = str(report_data.get("summary", ""))

        if report_sections:
            return f"# {title}\n\n{summary}\n\n{report_sections}"

        return f"# {title}\n\n{summary}"
