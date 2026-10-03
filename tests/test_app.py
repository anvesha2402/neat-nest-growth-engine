from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Home.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_copilot_page_answers_example():
    at = AppTest.from_file(str(ROOT / "pages/6_Nest_Copilot.py"), default_timeout=120).run()
    at.text_input[0].set_value("Why did sales decline in Hyderabad?")
    at.button[-1].click().run()
    assert not at.exception
    assert any("Hyderabad" in s.value for s in at.subheader)
