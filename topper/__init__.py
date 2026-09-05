"""TOPPER — top-only academic paper retrieval.

The venue tier gate runs before ranking; only papers from qualifying venues
are scored.

    from topper import search, SearchPolicy

    hits = search(
        "graph neural networks",
        policy=SearchPolicy(ccf_levels=("A", "B"), cas_zones=(1, 2)),
        limit=20,
    )
"""

from topper.models import PaperCard, SearchPolicy
from topper.pipeline import search
from topper.tiers.registry import get_default_registry

__all__ = ["PaperCard", "SearchPolicy", "search", "get_default_registry", "__version__"]
__version__ = "0.1.0"
