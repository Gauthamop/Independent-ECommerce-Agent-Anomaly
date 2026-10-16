"""Smoke tests for the version-2 app (LLM agent + real data). No test calls Groq."""
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "anomaly_agent" / "app" / "app_llm.py"
PAGES = ["Overview", "How data flows", "Explore metrics", "Detectors", "Agent (rules or Groq LLM)",
         "Evaluation (rules vs Groq)", "Real data (NAB)"]

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "clean" / "anomaly.duckdb").exists(),
                                reason="run make_raw, csv_adapter and run_compare first")


def open_page(page):
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=300).run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    open_page(page)


def test_an_unrelated_change_makes_the_rule_agent_confident_and_wrong():
    at = open_page("Agent (rules or Groq LLM)")
    incident = next(o for o in at.selectbox[0].options if "bank_c/south" in o)        # the fraud burst
    at.selectbox[0].set_value(incident).run()
    assert not at.exception
    assert any("Low" in m.value for m in at.markdown if m.value.startswith("**Confidence"))

    # selectboxes on this page: [0] incident, [1] hide a change, [2] add an unrelated change
    decoy = next(o for o in at.selectbox[2].options if o.startswith("search-service"))
    at.selectbox[2].set_value(decoy).run()
    assert not at.exception
    assert any("High" in m.value for m in at.markdown if m.value.startswith("**Confidence"))   # fooled by timing


def test_the_original_app_is_still_a_separate_entry_point():
    original = (ROOT / "src" / "anomaly_agent" / "app" / "app.py").read_text(encoding="utf-8")
    assert "app_llm" not in original and "Groq" not in original
