import itertools

import numpy as np
import pandas as pd

#for 30 days , seed=42 , and for a particular sale_date 
def make_orders(days=30, seed=42, sale_date="2025-01-15"):
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


def make_decline_rate(bank="bank_a", region="north", days=30, seed=42, base_rate=0.04):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=days * 288, freq="5min")

    # steady rate around 4% with small noise
    rate = rng.normal(base_rate, 0.005, size=len(ts))

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

#for 3 banks spread across 2 regions 
def make_all_decline_rates(days=30, seed=42):
    frames = []
    # 3 banks x 2 regions = 6 series; each gets its own seed so the noise differs
    for i, (bank, region) in enumerate(itertools.product(BANKS, REGIONS)):
        frames.append(make_decline_rate(bank, region, days=days, seed=seed + i))
    return pd.concat(frames, ignore_index=True)


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
    print(all_dr.groupby(["bank", "region"]).decline_rate.agg(["count", "mean", "std"]))
    plt.show()
