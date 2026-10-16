import duckdb
import pytest

from src.anomaly_agent.tools.metric_tools import MetricTools


def test_query_metric_is_small_and_correct(small_db):
    out = MetricTools(small_db).query_metric("m", "e1", "2025-01-03 12:00", "2025-01-03 14:00")
    assert out["n_points"] == 24
    assert len(out["series"]) <= 12
    assert 59 < out["mean"] < 61            # 50 + the +10 jump


def test_query_metric_empty_window(small_db):
    out = MetricTools(small_db).query_metric("m", "e1", "2030-01-01 00:00", "2030-01-02 00:00")
    assert out["n_points"] == 0 and "no data" in out["note"]


def test_compare_segments_finds_the_one_broken_series(small_db):
    out = MetricTools(small_db).compare_segments("m", "2025-01-03 12:00", "2025-01-03 14:00")
    assert out["top"][0]["entity"] == "e1"
    assert out["top"][0]["tags"] == {"group": "e1"}
    assert out["top"][0]["change_pct"] > 15
    assert out["n_changed"] == 1                     # only e1 moved
    assert out["rest_median_abs_change_pct"] < 3     # the others look normal


def test_get_recent_changes_closest_first_and_respects_lookback(small_db):
    tools = MetricTools(small_db)
    out = tools.get_recent_changes("2025-01-03 12:00", lookback_hours=48)
    assert [c["service"] for c in out["changes"]] == ["svc-a", "svc-b"]
    assert out["changes"][0]["minutes_before"] == 0
    narrow = tools.get_recent_changes("2025-01-03 12:00", lookback_hours=6)
    assert narrow["n_found"] == 1


def test_database_is_read_only(small_db):
    with pytest.raises(duckdb.Error):
        MetricTools(small_db)._query("CREATE TABLE hacked (a INTEGER)")


def test_sql_injection_text_is_just_a_value(small_db):
    tools = MetricTools(small_db)
    out = tools.query_metric("m", "e1'; DROP TABLE metrics; --", "2025-01-03 12:00", "2025-01-03 14:00")
    assert out["n_points"] == 0
    assert tools.query_metric("m", "e1", "2025-01-03 12:00", "2025-01-03 14:00")["n_points"] == 24
