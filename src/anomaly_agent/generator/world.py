import itertools
from pathlib import Path

import numpy as np
import pandas as pd

SALE_DATE = "2025-01-15"

# problem 3: short fraud-like burst at a QUIET hour (03:00-03:45, a Thursday).
# orders x1.6 and decline rate +11 points for ONE bank/region. No deploy causes it.
FRAUD_BURST = {
    "bank": "bank_c",
    "region": "south",
    "start": "2025-01-09 03:00",
    "end": "2025-01-09 03:45",
    "jump": 0.11,
    "orders_factor": 1.6,
}

#for 30 days , seed=42 , and for a particular sale_date
def make_orders(days=30, seed=42, sale_date=SALE_DATE, burst=FRAUD_BURST):
    rng = np.random.default_rng(seed)
    #factor=[monday,tuesday,wednesday,thursday,friday,saturday,sunday]
    factor = np.array([1, 1, 1, 1, 1, 0.8, 0.8])
    # one timestamp every 5 minutes: 30 days * 288 = 8640 rows
    ts = pd.date_range("2025-01-01", periods=days * 288, freq="5min")

    # hour of day as a decimal, e.g. 13:30 -> 13.5
    hour = ts.hour + ts.minute / 60

    # daily cycle: lowest around 4am, highest around 4pm
    base = 100
    cycle = 60 * np.sin((hour - 10) / 24 * 2 * np.pi)

    noise = rng.normal(0, 5, size=len(ts))

    orders = base + cycle + noise
    orders = orders * factor[ts.dayofweek]

    # sale day: orders x3 on one day. This is NOT an anomaly (a trap for detectors)
    is_sale = ts.date == pd.Timestamp(sale_date).date()
    orders = orders * np.where(is_sale, 3, 1)  #if its on that date , expect sales *3

    # fraud burst: a short window with more orders than normal
    if burst is not None:
        in_burst = (ts >= pd.Timestamp(burst["start"])) & (ts < pd.Timestamp(burst["end"]))
        orders = orders * np.where(in_burst, burst["orders_factor"], 1)

    return pd.DataFrame({"timestamp": ts, "orders": orders})


def make_decline_rate(bank="bank_a", region="north", days=30, seed=42, base_rate=0.04,
                      incident=None):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=days * 288, freq="5min")

    # steady rate around 4% with small noise
    rate = rng.normal(base_rate, 0.005, size=len(ts))

    # injected problem: add a jump only inside the incident window
    if incident is not None:
        start = pd.Timestamp(incident["start"])
        end = pd.Timestamp(incident["end"])
        in_window = (ts >= start) & (ts < end)
        rate = rate + np.where(in_window, incident["jump"], 0)

    # a rate is a share, so it must stay between 0 and 1
    rate = np.clip(rate, 0, 1)

    return pd.DataFrame({
        "timestamp": ts,
        "bank": bank,
        "region": region,
        "decline_rate": rate,
    })

BANKS = ["bank_a", "bank_b", "bank_c"]
REGIONS = ["north", "south"]

# problem 1: bad deploy breaks ONE bank in ONE region (4% -> 30% for 3 hours)
BAD_DEPLOY = {
    "bank": "bank_a",
    "region": "north",
    "service": "payment-gateway",   # the change that caused it (also in the deploy log)
    "start": "2025-01-20 14:00",
    "end": "2025-01-20 17:00",
    "jump": 0.26,
}


#for 3 banks spread across 2 regions
def make_all_decline_rates(days=30, seed=42, incidents=(BAD_DEPLOY, FRAUD_BURST)):
    frames = []
    # 3 banks x 2 regions = 6 series; each gets its own seed so the noise differs
    for i, (bank, region) in enumerate(itertools.product(BANKS, REGIONS)):
        # only the series that matches an incident gets it (none for the others)
        hit = next((x for x in incidents if x["bank"] == bank and x["region"] == region), None)
        frames.append(make_decline_rate(bank, region, days=days, seed=seed + i, incident=hit))
    return pd.concat(frames, ignore_index=True)


SERVERS = ["server_1", "server_2", "server_3"]

# problem 2: slow memory leak on ONE server (rises ~36 points over 8 days)
MEMORY_LEAK = {                   #we purposely created one where there is an increase of memory leak of 36
    "server": "server_2",
    "service": "checkout-service",  # the change that caused it (also in the deploy log)
    "start": "2025-01-23 00:00",
    "end": "2025-01-31 00:00",
    "rise": 36,
}


def make_memory(server="server_1", days=30, seed=42, leak=None):
    """Memory use in percent. A gauge: it goes up and down around a level."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=days * 288, freq="5min")

    hour = ts.hour + ts.minute / 60
    # small daily wiggle around 50%
    mem = 50 + 3 * np.sin((hour - 10) / 24 * 2 * np.pi) + rng.normal(0, 1, size=len(ts))

    if leak is not None:
        start = pd.Timestamp(leak["start"])
        end = pd.Timestamp(leak["end"])
        # 0 before the leak, then a straight climb to 1 at the end
        progress = np.clip((ts - start) / (end - start), 0, 1)
        mem = mem + leak["rise"] * np.asarray(progress)

    mem = np.clip(mem, 0, 100)
    return pd.DataFrame({"timestamp": ts, "server": server, "memory_pct": mem})


def make_all_memory(days=30, seed=42, leak=MEMORY_LEAK):
    frames = []
    for i, server in enumerate(SERVERS):
        hit = leak is not None and server == leak["server"]
        frames.append(make_memory(server, days=days, seed=seed + 100 + i,
                                  leak=leak if hit else None))
    return pd.concat(frames, ignore_index=True)


def make_latency(region="north", days=30, seed=42):
    """Response time in ms. A gauge: busier hours are a bit slower, and it has a long tail."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=days * 288, freq="5min")

    hour = ts.hour + ts.minute / 60
    # about 200 ms, ~20 ms slower at the afternoon peak
    level = 200 + 20 * np.sin((hour - 10) / 24 * 2 * np.pi)

    # lognormal noise: mostly small, sometimes a slow spike (never below 0)
    noise = rng.lognormal(mean=0, sigma=0.1, size=len(ts))
    return pd.DataFrame({"timestamp": ts, "region": region, "latency_ms": level * noise})


def make_all_latency(days=30, seed=42):
    frames = [make_latency(r, days=days, seed=seed + 200 + i) for i, r in enumerate(REGIONS)]
    return pd.concat(frames, ignore_index=True)


def make_deploy_log():
    """What a real company would have: what changed and when. It does NOT say who was hurt."""
    rows = [
        # harmless changes, so the agent can't just pick the only deploy
        ("2025-01-08 11:00", "search-service", "deploy", "v1.8 rollout, new ranking"),
        ("2025-01-13 10:30", "cart-service", "deploy", "v4.1 rollout, UI fixes"),
        # the bad one (same time as BAD_DEPLOY)
        (BAD_DEPLOY["start"], BAD_DEPLOY["service"], "deploy", "v2.3 rollout, new retry logic"),
        # a payment deploy that did NOT cause any problem
        ("2025-01-27 16:00", "payment-gateway", "deploy", "v2.4 rollout, logging change"),
        # the cause of the memory leak (same time as MEMORY_LEAK)
        (MEMORY_LEAK["start"], MEMORY_LEAK["service"], "config_change", "removed cache size limit"),
    ]
    log = pd.DataFrame(rows, columns=["timestamp", "service", "change_type", "description"])
    log["timestamp"] = pd.to_datetime(log["timestamp"])
    return log.sort_values("timestamp").reset_index(drop=True)


def make_ground_truth():
    """The answer key. Only the scoreboard may read this. Detectors never."""
    sale_start = pd.Timestamp(SALE_DATE)
    rows = [
        {
            "problem_id": 0,
            "type": "sale_day",
            "start": sale_start,
            "end": sale_start + pd.Timedelta(days=1),
            "metric": "orders",
            "series": "orders:all",
            "bank": None,
            "region": None,
            "is_anomaly": False,
            "true_cause": "planned sale (normal traffic spike)",
            "cause_type": "none",
        },
        {
            "problem_id": 1,
            "type": "bad_deploy",
            "start": pd.Timestamp(BAD_DEPLOY["start"]),
            "end": pd.Timestamp(BAD_DEPLOY["end"]),
            "metric": "decline_rate",
            "series": f"decline_rate:{BAD_DEPLOY['bank']}/{BAD_DEPLOY['region']}",
            "bank": BAD_DEPLOY["bank"],
            "region": BAD_DEPLOY["region"],
            "is_anomaly": True,
            "true_cause": "payment-gateway v2.3 deploy",
            "cause_type": "change",
            "cause_service": BAD_DEPLOY["service"],
            "cause_time": BAD_DEPLOY["start"],
        },
        {
            "problem_id": 2,
            "type": "memory_leak",
            "start": pd.Timestamp(MEMORY_LEAK["start"]),
            "end": pd.Timestamp(MEMORY_LEAK["end"]),
            "metric": "memory_pct",
            "series": f"memory_pct:{MEMORY_LEAK['server']}",
            "server": MEMORY_LEAK["server"],
            "is_anomaly": True,
            "true_cause": "checkout-service config change (cache limit removed)",
            "cause_type": "change",
            "cause_service": MEMORY_LEAK["service"],
            "cause_time": MEMORY_LEAK["start"],
        },
        {
            "problem_id": 3,
            "type": "fraud_burst",
            "start": pd.Timestamp(FRAUD_BURST["start"]),
            "end": pd.Timestamp(FRAUD_BURST["end"]),
            "metric": "orders+decline_rate",
            "series": f"orders:all;decline_rate:{FRAUD_BURST['bank']}/{FRAUD_BURST['region']}",
            "bank": FRAUD_BURST["bank"],
            "region": FRAUD_BURST["region"],
            "is_anomaly": True,
            "true_cause": "card-testing attack (external; no matching change in deploy log)",
            "cause_type": "external",
        },
    ]
    return pd.DataFrame(rows)


def write_records(root="data"):
    """Write only the answer key. The messy raw files come from make_raw.py."""
    root = Path(root)
    (root / "ground_truth").mkdir(parents=True, exist_ok=True)
    make_ground_truth().to_csv(root / "ground_truth" / "ground_truth.csv", index=False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = make_orders()
    print(df.shape)
    print(df.head())

    df.plot(x="timestamp", y="orders", figsize=(12, 4))

    dr = make_decline_rate()
    print(dr.shape)
    print(dr.head())
    print(dr.decline_rate.describe())

    dr.plot(x="timestamp", y="decline_rate", figsize=(12, 4))

    all_dr = make_all_decline_rates()
    print(all_dr.shape)
    print(all_dr.groupby(["bank", "region"]).decline_rate.agg(["count", "mean", "std", "max"]))

    # overall rate (simple average of the 6 series) vs the broken series
    overall = all_dr.groupby("timestamp").decline_rate.mean()
    broken = all_dr[(all_dr.bank == BAD_DEPLOY["bank"]) & (all_dr.region == BAD_DEPLOY["region"])]
    window = slice(BAD_DEPLOY["start"], BAD_DEPLOY["end"])
    print("overall max in window:", overall.loc[window].max().round(3))
    print("broken series max in window:", broken.set_index("timestamp").decline_rate.loc[window].max().round(3))

    lat = make_all_latency()
    print(lat.shape)
    print(lat.groupby("region").latency_ms.describe().round(1))

    mem = make_all_memory()
    print(mem.shape)
    print(mem.groupby("server").memory_pct.agg(["count", "mean", "max"]))
    leaky = mem[mem.server == MEMORY_LEAK["server"]].set_index("timestamp").memory_pct
    print("leaky server, daily mean:")
    print(leaky.resample("D").mean().round(1).loc["2025-01-21":].to_string())
    print("days over 80% for leaky server:", (leaky > 80).sum() / 288)

    # fraud burst check: orders in the window vs the same hour on other days
    o = df.set_index("timestamp").orders
    in_win = o.loc[FRAUD_BURST["start"]:FRAUD_BURST["end"]]
    print("burst orders (03:00-03:45):", in_win.mean().round(1))
    same_hour = o[(o.index.hour == 3) & (o.index.date != pd.Timestamp(FRAUD_BURST["start"]).date())]
    print("normal orders at 03:xx:", same_hour.mean().round(1), "| busiest orders overall:", o.drop(o.loc[SALE_DATE].index).max().round(1))
    fraud_series = all_dr[(all_dr.bank == FRAUD_BURST["bank"]) & (all_dr.region == FRAUD_BURST["region"])]
    print("bank_c/south max in burst:", fraud_series.set_index("timestamp").decline_rate.loc[FRAUD_BURST["start"]:FRAUD_BURST["end"]].max().round(3))

    write_records()
    print(make_deploy_log())
    print(make_ground_truth())

    ax = broken.plot(x="timestamp", y="decline_rate", figsize=(12, 4), label="bank_a/north")
    overall.plot(ax=ax, label="overall")
    ax.legend()
    plt.show()
