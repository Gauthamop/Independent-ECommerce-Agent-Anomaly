import json

import duckdb
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_db(tmp_path):
    """3 days, 5-minute points, metric "m" on entities e1/e2/e3.

    e1 jumps by +10 on day 3 from 12:00 to 14:00. Changes: svc-a at 12:00 on day 3 (the cause),
    svc-b a day earlier (a decoy). Day 1 is 2025-01-01.
    """
    rng = np.random.default_rng(1)
    ts = pd.date_range("2025-01-01", periods=3 * 288, freq="5min")
    frames = []
    for name in ["e1", "e2", "e3"]:
        values = 50 + rng.normal(0, 0.5, len(ts))
        if name == "e1":
            jump = (ts >= "2025-01-03 12:00") & (ts < "2025-01-03 14:00")
            values = values + np.where(jump, 10, 0)
        frames.append(pd.DataFrame({
            "timestamp": ts, "entity": name, "metric": "m", "value": values,
            "tags": json.dumps({"group": name}),
        }))
    metrics = pd.concat(frames, ignore_index=True)
    changes = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-02 08:00", "2025-01-03 12:00"]),
        "service": ["svc-b", "svc-a"],
        "change_type": ["deploy", "deploy"],
        "description": ["harmless", "the real cause"],
    })

    path = tmp_path / "small.duckdb"
    con = duckdb.connect(str(path))
    for name, frame in [("metrics", metrics), ("changes", changes)]:
        con.register("f", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM f")
        con.unregister("f")
    con.close()
    return str(path)
