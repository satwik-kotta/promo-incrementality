"""Promotion lift model and per deal economics.

Model: Poisson regression of weekly units on log net price, mailer feature, in store display,
week of deal, weeks after and before a deal, and the share of same category products on deal,
with product and department x week fixed effects. Standard errors are clustered by product.

The fitted coefficients give, for every deal week, what units would have been at the regular
price with no mailer and no display change. The gap is the incremental lift. Each deal is then
costed: extra discount given to people who would have bought anyway, margin earned on the
incremental units, and margin lost on same category products the deal pulled sales from.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

DEAL_THRESHOLD = 0.15  # a week is a deal week when the loyalty discount is at least 15% of shelf price
FORMULA = ("units ~ log_price + mailer_ad + display + sibling_promo_share + deal_w2 + deal_w3plus"
           " + post_1 + post_2 + post_3 + post_4 + pre_1 + pre_2 | product_id + dept_week")
MARGINS = [0.20, 0.25, 0.30, 0.35, 0.40]


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    from features import add_episode_features, add_sibling_promo

    p = panel.copy()
    p["mailer_ad"] = p["mailer"]
    p["depth"] = np.where(p["gross_revenue"] > 0, p["loyalty_discount"] / p["gross_revenue"], np.nan)
    # Weeks with no sales have no observed price. Fill with the product's typical depth for that ad state.
    typical_off = p[p["mailer_ad"] == 0].groupby("product_id")["depth"].median()
    typical_ad = p[p["mailer_ad"] == 1].groupby("product_id")["depth"].mean()
    fill = np.where(p["mailer_ad"] == 1, p["product_id"].map(typical_ad), p["product_id"].map(typical_off))
    p["depth_obs"] = p["depth"].notna()
    p["depth"] = p["depth"].fillna(pd.Series(fill, index=p.index)).fillna(0).clip(0, 0.9)
    p["log_price"] = np.log(1 - p["depth"])
    # Episode code works on a 0/1 column named mailer, so point it at the deal flag.
    p["mailer"] = (p["depth"] >= DEAL_THRESHOLD).astype(int)
    df = add_sibling_promo(add_episode_features(p))
    df = df.rename(columns={"mailer": "deal", "promo_w2": "deal_w2", "promo_w3plus": "deal_w3plus"})
    df["sibling_promo_share"] = (
        df.groupby(["product_category", "week"])["deal"].transform("sum") - df["deal"]
    ) / (df["n_in_category"] - 1).clip(lower=1)
    # Regular (non deal) depth and shelf price for each product.
    base = df[df["deal"] == 0]
    df["base_depth"] = df["product_id"].map(base.groupby("product_id")["depth"].median()).fillna(0)
    shelf = (base["gross_revenue"] / base["units"].where(base["units"] > 0)).groupby(base["product_id"]).median()
    df["shelf_price"] = df["product_id"].map(shelf)
    return df


def fit(df: pd.DataFrame):
    return pf.fepois(FORMULA, data=df, vcov={"CRV1": "product_id"})


def effects_table(model) -> pd.DataFrame:
    t = model.tidy().reset_index().rename(columns={"Coefficient": "term"})
    t["effect_pct"] = (np.exp(t["Estimate"]) - 1) * 100
    t["ci_low_pct"] = (np.exp(t["2.5%"]) - 1) * 100
    t["ci_high_pct"] = (np.exp(t["97.5%"]) - 1) * 100
    return t[["term", "Estimate", "Std. Error", "Pr(>|t|)", "effect_pct", "ci_low_pct", "ci_high_pct"]]


def deal_economics(df: pd.DataFrame, coef: dict) -> pd.DataFrame:
    """Cost each deal episode. Returns one row per episode."""
    d = df[df["deal"] == 1].copy()
    # Log change in expected units caused by the deal itself, against the same week at regular price.
    lift_log = (coef["log_price"] * (d["log_price"] - np.log(1 - d["base_depth"]))
                + coef["mailer_ad"] * d["mailer_ad"]
                + coef["deal_w2"] * d["deal_w2"] + coef["deal_w3plus"] * d["deal_w3plus"])
    d["cf_units"] = d["units"] * np.exp(-lift_log)
    d["incr_units"] = d["units"] - d["cf_units"]
    d["net_rev"] = d["gross_revenue"] - d["loyalty_discount"]
    d["cf_net_rev"] = d["cf_units"] * d["shelf_price"] * (1 - d["base_depth"])

    # Cannibalization: units this deal took from other products in the same category that week.
    others = df[["product_id", "product_category", "week", "units", "n_in_category", "shelf_price", "base_depth"]]
    cat_week = others.groupby(["product_category", "week"]).agg(
        cat_units=("units", "sum"),
        cat_rev=("units", lambda s: 0.0),
    )
    rev = (others["units"] * others["shelf_price"] * (1 - others["base_depth"])).groupby(
        [others["product_category"], others["week"]]).sum()
    cat_week["cat_net_rev_regular"] = rev
    d = d.join(cat_week, on=["product_category", "week"])
    share_one = 1 / (d["n_in_category"] - 1).clip(lower=1)
    lost_frac = 1 - np.exp(coef["sibling_promo_share"] * share_one)
    other_units = d["cat_units"] - d["units"]
    other_rev = d["cat_net_rev_regular"] - d["units"] * d["shelf_price"] * (1 - d["base_depth"])
    d["cannibal_units"] = (other_units * lost_frac).clip(lower=0)
    d["cannibal_rev"] = (other_rev * lost_frac).clip(lower=0)

    ep = d.groupby(["product_id", "episode_id"]).agg(
        department=("department", "first"),
        product_category=("product_category", "first"),
        weeks=("week", "count"),
        start_week=("week", "min"),
        mailer=("mailer_ad", "max"),
        depth=("depth", "mean"),
        base_depth=("base_depth", "first"),
        shelf_price=("shelf_price", "first"),
        units=("units", "sum"),
        cf_units=("cf_units", "sum"),
        incr_units=("incr_units", "sum"),
        net_rev=("net_rev", "sum"),
        cf_net_rev=("cf_net_rev", "sum"),
        cannibal_units=("cannibal_units", "sum"),
        cannibal_rev=("cannibal_rev", "sum"),
    ).reset_index()
    # Profit change at gross margin m (margin on the regular shelf price):
    #   unit cost c = (1 - m) * shelf price
    #   own change   = (net_rev - c * units) - (cf_net_rev - c * cf_units)
    #   cannibal     = m_net * cannibal_rev, using the same margin on the regular net price
    # Break even margin solves own change - cannibal = 0, which is linear in m.
    p = ep["shelf_price"]
    a = ep["net_rev"] - ep["cf_net_rev"] - p * (ep["units"] - ep["cf_units"])
    b = p * (ep["units"] - ep["cf_units"])
    # profit(m) = a + m * b - cannibal_margin(m); cannibal margin ~ m * cannibal_rev
    slope = b - ep["cannibal_rev"]
    ep["break_even_margin"] = np.where(slope > 0, -a / slope, np.nan)
    for m in MARGINS:
        ep[f"profit_at_{int(m * 100)}"] = a + m * slope
    ep["subsidized_share"] = ep["cf_units"] / ep["units"].where(ep["units"] > 0)
    ep["discount_dollars"] = (ep["depth"] - ep["base_depth"]).clip(lower=0) * p * ep["units"]
    return ep


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    OUT.mkdir(exist_ok=True)
    panel = pd.read_parquet(PROC / "panel.parquet")
    df = prepare(panel)
    model = fit(df)
    eff = effects_table(model)
    coef = dict(zip(eff["term"], eff["Estimate"]))
    ep = deal_economics(df, coef)
    df.to_parquet(PROC / "panel_features.parquet")
    ep.to_parquet(PROC / "deal_episodes.parquet")
    eff.to_csv(OUT / "lift_effects.csv", index=False)
    ep.to_csv(OUT / "deal_episodes.csv", index=False)
    print(eff.round(3).to_string())
    print(ep.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
