import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import add_episode_features  # noqa: E402
import model  # noqa: E402


def toy_panel(flags):
    n = len(flags)
    return pl.DataFrame({
        "product_id": ["p"] * n, "week": list(range(2, 2 + n)), "department": ["GROCERY"] * n,
        "product_category": ["X"] * n, "mailer": flags, "display_share": [0.0] * n,
    })


def test_episodes_and_event_time():
    df = add_episode_features(toy_panel([0, 0, 1, 1, 0, 0, 0, 0, 0]))
    assert df["promo_week_no"].to_list() == [0, 0, 1, 2, 0, 0, 0, 0, 0]
    assert df["weeks_after"].to_list() == [0, 0, 0, 0, 1, 2, 3, 4, 0]
    assert df["weeks_before"].to_list() == [2, 1, 0, 0, 0, 0, 0, 0, 0]


def test_back_to_back_deals_do_not_count_as_after_weeks():
    df = add_episode_features(toy_panel([1, 0, 1, 0, 0]))
    assert df["episode_id"].to_list() == [0, -1, 1, -1, -1]
    assert df["weeks_after"][1] == 1
    assert df["weeks_after"][3] == 1


def simulated_panel(seed=0, elasticity=-1.8, mailer_lift=0.4):
    rng = np.random.default_rng(seed)
    rows = []
    cats = ["A", "B", "C", "D"]
    for p in range(120):
        level = rng.uniform(5, 30)
        price = rng.uniform(1, 5)
        for w in range(2, 53):
            deal = rng.random() < 0.3
            depth = rng.uniform(0.15, 0.4) if deal else 0.0
            ad = int(deal and rng.random() < 0.6)
            mu = level * np.exp(elasticity * np.log(1 - depth) + mailer_lift * ad)
            units = rng.poisson(mu)
            rows.append({
                "product_id": f"p{p}", "week": w, "department": "GROCERY", "product_category": cats[p % 4],
                "units": units, "gross_revenue": units * price, "loyalty_discount": units * price * depth,
                "buyers": units, "mailer": ad, "display_share": 0.0,
            })
    return pl.DataFrame(rows)


def test_model_recovers_known_price_and_mailer_effects():
    df = model.prepare(simulated_panel())
    fit = model.fit(df)
    coef = dict(zip(model.effects_table(fit)["term"], model.effects_table(fit)["Estimate"]))
    assert abs(coef["log_price"] - (-1.8)) < 0.2
    assert abs(coef["mailer_ad"] - 0.4) < 0.1
    # No dip or pre trend was simulated, so these should be near zero.
    for t in ["post_2", "post_3", "pre_2"]:
        assert abs(coef[t]) < 0.1


def test_profit_is_zero_at_break_even_margin():
    df = model.prepare(simulated_panel(seed=1))
    coef = dict(zip(model.effects_table(model.fit(df))["term"], model.effects_table(model.fit(df))["Estimate"]))
    ep = model.deal_economics(df, coef).drop_nulls(["break_even_margin", "shelf_price"])
    row = ep.row(0, named=True)
    p = row["shelf_price"]
    m = row["break_even_margin"]
    a = row["net_rev"] - row["cf_net_rev"] - p * (row["units"] - row["cf_units"])
    slope = p * (row["units"] - row["cf_units"]) - row["cannibal_rev"]
    assert abs(a + m * slope) < 1e-6
    assert ((ep["cf_units"] <= ep["units"] + 1e-9) | (ep["units"] == 0)).all()
