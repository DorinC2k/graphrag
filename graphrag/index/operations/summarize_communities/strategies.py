# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run, _run_extractor and _load_nodes_edges_for_claim_chain methods definition."""

import logging
import traceback

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.index.operations.summarize_communities.community_reports_extractor import (
    CommunityReportsExtractor,
)
from graphrag.index.operations.summarize_communities.typing import (
    CommunityReport,
    Finding,
    StrategyConfig,
)
from graphrag.index.utils.rate_limiter import RateLimiter
from graphrag.language_model.manager import ModelManager
from graphrag.language_model.protocol.base import ChatModel

log = logging.getLogger(__name__)


async def run_graph_intelligence(
    community: str | int,
    input: str,
    level: int,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    args: StrategyConfig,
) -> CommunityReport | None:
    """Run the graph intelligence entity extraction strategy."""
    llm_config = LanguageModelConfig(**args["llm"])
    llm = ModelManager().get_or_create_chat_model(
        name="community_reporting",
        model_type=llm_config.type,
        config=llm_config,
        callbacks=callbacks,
        cache=cache,
    )

    return await _run_extractor(llm, community, input, level, args, callbacks)


async def _run_extractor(
    model: ChatModel,
    community: str | int,
    input: str,
    level: int,
    args: StrategyConfig,
    callbacks: WorkflowCallbacks,
) -> CommunityReport | None:
    # RateLimiter
    rate_limiter = RateLimiter(rate=1, per=60)
    extractor = CommunityReportsExtractor(
        model,
        extraction_prompt=args.get("extraction_prompt", None),
        max_report_length=args.get("max_report_length", None),
        on_error=lambda e, stack, _data: callbacks.error(
            "Community Report Extraction Error", e, stack
        ),
    )

    try:
        await rate_limiter.acquire()
        results = await extractor(input)
        report = results.structured_output
        if report is None:
            log.warning("No report found for community: %s", community)
            return None

        report_dict = CommunityReportsExtractor._model_dump(report)
        findings = CommunityReportsExtractor._normalize_findings(
            report_dict.get("findings")
        )

        rating = report_dict.get("rating")
        try:
            rank = float(rating) if rating is not None else 0.0
        except (TypeError, ValueError):
            rank = 0.0

        return CommunityReport(
            community=community,
            full_content=results.output,
            level=level,
            rank=rank,
            title=str(report_dict.get("title", "")),
            rating_explanation=str(report_dict.get("rating_explanation", "")),
            summary=str(report_dict.get("summary", "")),
            findings=[
                Finding(explanation=finding["explanation"], summary=finding["summary"])
                for finding in findings
            ],
            full_content_json=CommunityReportsExtractor._model_dump_json(
                report, indent=4
            ),
        )
    except Exception as e:
        log.exception("Error processing community: %s", community)
        callbacks.error("Community Report Extraction Error", e, traceback.format_exc())
        return None
