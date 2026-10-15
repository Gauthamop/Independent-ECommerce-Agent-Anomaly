"""Turn the clean generated world into deliberately MESSY raw files.

Messy on purpose: odd column names, a different time format in every file,
wide format (one column per series), and local time (IST) instead of UTC.
The adapter has to clean all of this. The answer key stays in UTC.
"""
from pathlib import Path

import pandas as pd

from src.anomaly_agent.generator.world import (
    make_all_decline_rates,
    make_all_latency,
    make_all_memory,
    make_deploy_log,
    make_orders,
    write_records,
)

LOCAL_TZ = "Asia/Kolkata"


def to_local(ts, fmt):
    """The generator's naive timestamps are UTC. Show them as local-time text."""
    local = pd.DatetimeIndex(ts).tz_localize("UTC").tz_convert(LOCAL_TZ)
    return local.strftime(fmt)


def write_raw(root="data"):
    root = Path(root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # 1. orders: one metric, long-ish file, day-first dates
    orders = make_orders()
    pd.DataFrame({
        "ts_local": to_local(orders.timestamp, "%d/%m/%Y %H:%M"),
        "ord_cnt": orders.orders,
    }).to_csv(raw / "orders_export.csv", index=False)

    # 2. payments: WIDE, one column per bank x region, column name has spaces
    dr = make_all_decline_rates()
    wide = dr.pivot(index="timestamp", columns=["bank", "region"], values="decline_rate")
    wide.columns = [f"dclnrt_{bank}_{region}" for bank, region in wide.columns]
    wide.insert(0, "Time (IST)", to_local(wide.index, "%Y-%m-%d %H:%M:%S"))
    wide.to_csv(raw / "payments_wide.csv", index=False)

    # 3. infra: WIDE, latency and memory mixed in one file, month names in dates
    lat = make_all_latency().pivot(index="timestamp", columns="region", values="latency_ms")
    lat.columns = [f"lat_ms_{c}" for c in lat.columns]
    mem = make_all_memory().pivot(index="timestamp", columns="server", values="memory_pct")
    mem.columns = [f"mem_pct_{c}" for c in mem.columns]
    infra = lat.join(mem)
    infra.insert(0, "time_ist", to_local(infra.index, "%d-%b-%Y %H:%M"))
    infra.to_csv(raw / "infra_wide.csv", index=False)

    # 4. change records: odd column names, local time
    log = make_deploy_log()
    pd.DataFrame({
        "when_local": to_local(log.timestamp, "%Y-%m-%d %H:%M"),
        "svc": log.service,
        "kind": log.change_type,
        "note": log.description,
    }).to_csv(raw / "deploy_log.csv", index=False)

    # the answer key (UTC), kept far away from the raw files
    write_records(root)


if __name__ == "__main__":
    write_raw()
    for f in sorted(Path("data").rglob("*.csv")):
        print(f, f.stat().st_size // 1024, "KB")
