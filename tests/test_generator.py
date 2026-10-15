import pandas as pd

from src.anomaly_agent.generator.world import (
    SALE_DATE,
    make_all_decline_rates,
    make_ground_truth,
    make_orders,
)


def test_orders_shape_and_sale_day():
    df = make_orders()
    assert len(df) == 30 * 288
    daily_peak = df.groupby(df.timestamp.dt.date).orders.max()
    sale = daily_peak[pd.Timestamp(SALE_DATE).date()]
    assert sale > 2.5 * daily_peak.drop(pd.Timestamp(SALE_DATE).date()).median()


def test_only_the_target_series_is_broken():
    dr = make_all_decline_rates()
    assert dr.groupby(["bank", "region"]).ngroups == 6
    peaks = dr.groupby(["bank", "region"]).decline_rate.max()
    assert peaks[("bank_a", "north")] > 0.25
    assert peaks[("bank_c", "south")] > 0.12          # the fraud burst
    assert peaks.drop([("bank_a", "north"), ("bank_c", "south")]).max() < 0.07


def test_ground_truth_marks_sale_day_as_normal():
    truth = make_ground_truth().set_index("type")
    assert not truth.loc["sale_day", "is_anomaly"]
    assert truth.loc[["bad_deploy", "memory_leak", "fraud_burst"], "is_anomaly"].all()
