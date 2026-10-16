"""Smoke test: every page of the Streamlit app renders without an error (needs the built data)."""
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "anomaly_agent" / "app" / "app.py"
PAGES = ["Overview", "How data flows", "Explore metrics", "Detectors", "Agent", "Evaluation"]

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "clean" / "anomaly.duckdb").exists(),
                                reason="run make_raw, csv_adapter and run_compare first")


@pytest.mark.parametrize("metric", ["decline_rate", "latency_ms", "memory_pct", "orders"])
@pytest.mark.parametrize("resolution", ["5 min", "15 min", "1 hour"])
def test_explore_every_metric_and_resolution(metric, resolution):
    """Regression: resampling broke for any selection whose rows were not 0,1,2... in the table."""
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.radio[0].set_value("Explore metrics").run()
    at.selectbox[0].set_value(metric).run()
    at.multiselect[0].set_value(list(at.multiselect[0].options)).run()   # all series
    at.selectbox[1].set_value(resolution).run()
    assert not at.exception, f"{metric} / {resolution}: {at.exception}"


def test_orders_alerts_explained_and_threshold_alerts_available():
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=300).run()
    at.sidebar.radio[0].set_value("Explore metrics").run()
    at.selectbox[0].set_value("orders").run()

    # the final system raised nothing on orders: the page must say so instead of looking broken
    assert not at.exception
    assert any("No alerts from" in i.value for i in at.info)

    # the simple threshold guard did raise (false) alarms on the sale day: no message, no crash
    at.selectbox[2].set_value("Threshold only (hand-set limits)").run()
    assert not at.exception
    assert not any("No alerts from" in i.value for i in at.info)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"
