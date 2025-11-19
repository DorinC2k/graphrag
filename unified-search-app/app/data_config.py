# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Data config module."""

# This file is used to store configurations for the graph-indexed data and the LLM/embeddings models used in the app.

# name of the table in the graph-indexed data where the communities are stored
communities_table = "output/communities"

# name of the table in the graph-indexed data where the community reports are stored
community_report_table = "output/community_reports"
# fallback table names to try if the primary community report table is missing
community_report_table_fallbacks = [
    community_report_table,
    # allow pointing the DATA_ROOT directly at the output directory
    "community_reports",
    # legacy name used by older pipeline runs
    "output/create_final_community_reports",
    "create_final_community_reports",
]

# name of the table in the graph-indexed data where the entity embeddings are stored
entity_table = "output/entities"

# name of the table in the graph-indexed data where the entity relationships are stored
relationship_table = "output/relationships"

# name of the table in the graph-indexed data where the entity covariates are stored
covariate_table = "output/covariates"

# name of the table in the graph-indexed data where the text units are stored
text_unit_table = "output/text_units"

# default configurations for LLM's answer generation, used in all search types
# this should be adjusted based on the token limits of the LLM model being used
# The following setting is for gpt-4-1106-preview (i.e. gpt-4-turbo)
# For gpt-4 (token-limit = 8k), a good setting could be:
default_suggested_questions = 5

# default timeout for streamlit cache
default_ttl = 60 * 60 * 24 * 7
