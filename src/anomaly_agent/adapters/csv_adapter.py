"""Messy raw files -> clean DuckDB tables, driven only by the profile YAML.

Tables written:
  metrics(timestamp, entity, metric, value, tags)   timestamps are UTC
  metric_info(metric, type)                         rate / count / gauge
  changes(timestamp, service, change_type, description)
"""
import json
import re
from pathlib import Path

import duckdb
import pandas as pd
import yaml


def load_profile(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_utc(text_col, fmt, tz):
    """Local-time text -> naive UTC timestamps."""
    local = pd.to_datetime(text_col, format=fmt).dt.tz_localize(tz)
    return local.dt.tz_convert("UTC").dt.tz_localize(None)


def read_metric_source(src, raw_dir, tz):
    df = pd.read_csv(Path(raw_dir) / src["file"])
    ts = to_utc(df[src["time_col"]], src["time_format"], tz)

    parts, used = [], {src["time_col"]}
    for rule in src["columns"]:
        pattern = re.compile(rule["pattern"])
        for col in df.columns:
            m = pattern.match(col)
            if not m:
                continue
            groups = m.groupdict()
            tags = {name: groups[name] for name in rule.get("tags", [])}
            parts.append(pd.DataFrame({
                "timestamp": ts,
                "entity": rule["entity"].format(**groups),
                "metric": rule["metric"],
                "value": df[col].astype(float),
                "tags": json.dumps(tags),
            }))
            used.add(col)

    # a column nobody described is a data-quality warning, not a silent skip
    unused = [c for c in df.columns if c not in used]
    if unused:
        print(f"WARNING {src['file']}: columns not in profile: {unused}")
    return pd.concat(parts, ignore_index=True)


def read_changes(spec, raw_dir, tz):
    df = pd.read_csv(Path(raw_dir) / spec["file"])
    out = pd.DataFrame({"timestamp": to_utc(df[spec["time_col"]], spec["time_format"], tz)})
    for clean_name, raw_name in spec["columns"].items():
        out[clean_name] = df[raw_name]
    return out.sort_values("timestamp").reset_index(drop=True)


def build(profile_path="profiles/ecommerce.yaml", raw_dir="data/raw",
          db_path="data/clean/anomaly.duckdb"):
    profile = load_profile(profile_path)
    tz = profile["timezone"]

    metrics = pd.concat(
        [read_metric_source(s, raw_dir, tz) for s in profile["sources"]],
        ignore_index=True,
    ).dropna(subset=["value"])

    # real exports repeat timestamps: average them, and say so (never drop data silently)
    key = ["metric", "entity", "timestamp"]
    n_dup = int(metrics.duplicated(key).sum())
    if n_dup:
        print(f"WARNING: {n_dup} duplicate timestamps were averaged")
        metrics = metrics.groupby(key, as_index=False).agg(value=("value", "mean"), tags=("tags", "first"))
    metrics = metrics.sort_values(key).reset_index(drop=True)[["timestamp", "entity", "metric", "value", "tags"]]

    metric_info = pd.DataFrame(
        [{"metric": name, "type": spec["type"], "description": spec.get("description", "")}
         for name, spec in profile["metrics"].items()]
    )
    changes = read_changes(profile["changes"], raw_dir, tz)
    service_info = pd.DataFrame(
        [{"service": name, "description": text} for name, text in profile.get("services", {}).items()],
        columns=["service", "description"],
    ).astype(str)

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    for name, frame in [("metrics", metrics), ("metric_info", metric_info), ("changes", changes),
                        ("service_info", service_info)]:
        con.register("frame", frame)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM frame")
        con.unregister("frame")
    con.close()
    return {"metrics": len(metrics), "metric_info": len(metric_info), "changes": len(changes)}


if __name__ == "__main__":
    print(build())
