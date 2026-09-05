"""Point the whole suite at synthetic tier tables.

The real CAS / FMS catalogues are licensed and not redistributed, so tests must
not depend on them — otherwise the suite is red on a fresh clone.
"""

import os
from pathlib import Path

import pytest

FIXTURE_DATA = Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(autouse=True, scope="session")
def _fixture_tier_tables():
    os.environ["TOPPER_DATA_DIR"] = str(FIXTURE_DATA)
    from topper.policy import _flagship_keys
    from topper.tiers.registry import get_default_registry

    get_default_registry.cache_clear()
    _flagship_keys.cache_clear()
    yield
