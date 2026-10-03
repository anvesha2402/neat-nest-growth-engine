import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session", autouse=True)
def built():
    """Build data, models and reports once per test session if they are missing."""
    from neatnest.config import MODEL_DIR, REPORT_DIR
    if not (REPORT_DIR / "results.json").exists() or not (MODEL_DIR / "lapse.joblib").exists():
        from neatnest.pipeline import run
        run(make_figures=False)
    return True
