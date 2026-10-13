import itertools
from pathlib import Path

import numpy as np
import pandas as pd

SALE_DATE = "2025-01-15"

#for 30 days , seed=42 , and for a particular sale_date
def make_orders(days=30, seed=42, sale_date=SALE_DATE):
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
    "start": "2025-01-20 14:00",
    "end": "2025-01-20 17:00",
    "jump": 0.26,
}


#for 3 banks spread across 2 regions
def make_all_decline_rates(days=30, seed=42, incident=BAD_DEPLOY):
    frames = []
    # 3 banks x 2 regions = 6 series; each gets its own seed so the noise differs
    for i, (bank, region) in enumerate(itertools.product(BANKS, REGIONS)):
        # only the series that matches the incident gets it
        hit = incident is not None and bank == incident["bank"] and region == incident["region"]
        frames.append(make_decline_rate(bank, region, days=days, seed=seed + i,
                                        incident=incident if hit else None))
    return pd.concat(frames, ignore_index=True)


def make_deploy_log():
    """What a real company would have: what changed and when. It does NOT say who was hurt."""
    rows = [
        # harmless changes, so the agent can't just pick the only deploy
        ("2025-01-08 11:00", "search-service", "deploy", "v1.8 rollout, new ranking"),
        ("2025-01-13 10:30", "cart-service", "deploy", "v4.1 rollout, UI fixes"),
        # the bad one (same time as BAD_DEPLOY)
        (BAD_DEPLOY["start"], "payment-gateway", "deploy", "v2.3 rollout, new retry logic"),
        # a payment deploy that did NOT cause any problem
        ("2025-01-27 16:00", "payment-gateway", "deploy", "v2.4 rollout, logging change"),
    ]
    log = pd.DataFrame(rows, columns=["timestamp", "service", "change_type", "description"])
    log["timestamp"] = pd.to_datetime(log["timestamp"])
    return log


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
            "bank": None,
            "region": None,
            "is_anomaly": False,
            "true_cause": "planned sale (normal traffic spike)",
        },
        {
            "problem_id": 1,
            "type": "bad_deploy",
            "start": pd.Timestamp(BAD_DEPLOY["start"]),
            "end": pd.Timestamp(BAD_DEPLOY["end"]),
            "metric": "decline_rate",
            "bank": BAD_DEPLOY["bank"],
            "region": BAD_DEPLOY["region"],
            "is_anomaly": True,
            "true_cause": "payment-gateway v2.3 deploy",
        },
    ]
    return pd.DataFrame(rows)


def write_records(root="data"):
    root = Path(root)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "ground_truth").mkdir(parents=True, exist_ok=True)

    make_deploy_log().to_csv(root / "raw" / "deploy_log.csv", index=False)
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

    write_records()
    print(make_deploy_log())
    print(make_ground_truth())

    ax = broken.plot(x="timestamp", y="decline_rate", figsize=(12, 4), label="bank_a/north")
    overall.plot(ax=ax, label="overall")
    ax.legend()
    plt.show()
