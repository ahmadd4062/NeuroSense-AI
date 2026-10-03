"""
NeuroSense AI Recommendation RAG package.

This package contains:
- rag_service: personalized recommendation retrieval
- resource_selector: verified YouTube/resource matching
- validator: knowledge-base consistency checks
"""

from .rag_service import (
    REPORT_SECTIONS,
    SECTION_LABELS,
    build_index,
    retrieve_report_recommendations,
    format_retrieved_context,
)

from .resource_selector import (
    select_report_resources,
    inject_resources_into_report,
)

__all__ = [
    "REPORT_SECTIONS",
    "SECTION_LABELS",
    "build_index",
    "retrieve_report_recommendations",
    "format_retrieved_context",
    "select_report_resources",
    "inject_resources_into_report",
]
