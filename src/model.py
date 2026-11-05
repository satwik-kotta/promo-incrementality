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
import polars as pl
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

DEAL_THRESHOLD = 0.15  # a week is a deal week when the loyalty discount is at least 15% of shelf price
FORMULA = ("units ~ log_price + mailer_ad + display + sibling_promo_share + deal_w2 + deal_w3plus"
           " + post_1 + post_2 + post_3 + post_4 + pre_1 + pre_2 | product_id + dept_week")
MARGINS = [0.20, 0.25, 0.30, 0.35, 0.40]


def prepare(panel: pl.DataFrame) -> pl.DataFrame:
    from features import add_episode_features, add_sibling_promo

    p = panel.with_columns(
        pl.col("mailer").alias("mailer_ad"),
        pl.when(pl.col("gross_revenue") > 0)
        .then(pl.col("loyalty_discount") / pl.col("gross_revenue")).alias("depth"),
    )
    # Weeks with no sales have no observed price. Fill with the product's typical depth for that ad state.
    typical_off = p.filter(pl.col("mailer_ad") == 0).group_by("product_id").agg(pl.col("depth").median().alias("_off"))
    typical_ad = p.filter(pl.col("mailer_ad") == 1).group_by("product_id").agg(pl.col("depth").mean().alias("_ad"))
    p = (p.join(typical_off, on="product_id", how="left", maintain_order="left")
          .join(typical_ad, on="product_id", how="left", maintain_order="left"))
    fill = pl.when(pl.col("mailer_ad") == 1).then(pl.col("_ad")).otherwise(pl.col("_off"))
    p = p.with_columns(
        pl.col("depth").is_not_null().alias("depth_obs"),
        pl.coalesce(pl.col("depth"), fill, pl.lit(0.0)).clip(0, 0.9).alias("depth"),
    ).drop("_off", "_ad")
    p = p.with_columns(
        (1 - pl.col("depth")).log().alias("log_price"),
        # Episode code works on a 0/1 column named mailer, so point it at the deal flag.
        (pl.col("depth") >= DEAL_THRESHOLD).cast(pl.Int64).alias("mailer"),
    )
    df = add_sibling_promo(add_episode_features(p))
    df = df.rename({"mailer": "deal", "promo_w2": "deal_w2", "promo_w3plus": "deal_w3plus"})
    df = df.with_columns(
        ((pl.col("deal").sum().over(["product_category", "week"]) - pl.col("deal"))
         / (pl.col("n_in_category") - 1).clip(lower_bound=1)).alias("sibling_promo_share")
    )
    # Regular (non deal) depth and shelf price for each product.
    base = df.filter(pl.col("deal") == 0).group_by("product_id").agg(
        pl.col("depth").median().alias("base_depth"),
        pl.when(pl.col("units") > 0).then(pl.col("gross_revenue") / pl.col("units")).median().alias("shelf_price"),
    )
    df = df.join(base, on="product_id", how="left", maintain_order="left")
    return df.with_columns(pl.col("base_depth").fill_null(0.0))


def fit(df: pl.DataFrame):
    # pyfixest takes a pandas frame, so convert only at the model boundary.
    return pf.fepois(FORMULA, data=df.to_pandas(), vcov={"CRV1": "product_id"})


def effects_table(model) -> pl.DataFrame:
    t = pl.from_pandas(model.tidy().reset_index()).rename({"Coefficient": "term"})
    pct = lambda c: ((pl.col(c).exp() - 1) * 100)  # noqa: E731
    return t.with_columns(
        pct("Estimate").alias("effect_pct"), pct("2.5%").alias("ci_low_pct"), pct("97.5%").alias("ci_high_pct"),
    ).select(["term", "Estimate", "Std. Error", "Pr(>|t|)", "effect_pct", "ci_low_pct", "ci_high_pct"])


def deal_economics(df: pl.DataFrame, coef: dict) -> pl.DataFrame:
    """Cost each deal episode. Returns one row per episode."""
    c = pl.col
    regular_price = c("shelf_price") * (1 - c("base_depth"))
    # Log change in expected units caused by the deal itself, against the same week at regular price.
    lift_log = (coef["log_price"] * (c("log_price") - (1 - c("base_depth")).log())
                + coef["mailer_ad"] * c("mailer_ad")
                + coef["deal_w2"] * c("deal_w2") + coef["deal_w3plus"] * c("deal_w3plus"))
    d = df.filter(c("deal") == 1).with_columns((c("units") * (-lift_log).exp()).alias("cf_units"))
    d = d.with_columns(
        (c("units") - c("cf_units")).alias("incr_units"),
        (c("gross_revenue") - c("loyalty_discount")).alias("net_rev"),
        (c("cf_units") * regular_price).alias("cf_net_rev"),
    )

    # Cannibalization: units this deal took from other products in the same category that week.
    cat_week = df.group_by(["product_category", "week"]).agg(
        c("units").sum().alias("cat_units"),
        (c("units") * regular_price).sum().alias("cat_net_rev_regular"),
    )
    d = d.join(cat_week, on=["product_category", "week"], how="left", maintain_order="left")
    lost_frac = 1 - (coef["sibling_promo_share"] / (c("n_in_category") - 1).clip(lower_bound=1)).exp()
    d = d.with_columns(
        ((c("cat_units") - c("units")) * lost_frac).clip(lower_bound=0).alias("cannibal_units"),
        ((c("cat_net_rev_regular") - c("units") * regular_price) * lost_frac).clip(lower_bound=0).alias("cannibal_rev"),
    )

    ep = d.group_by(["product_id", "episode_id"]).agg(
        c("department").drop_nulls().first(),
        c("product_category").drop_nulls().first(),
        c("week").count().alias("weeks"),
        c("week").min().alias("start_week"),
        c("mailer_ad").max().alias("mailer"),
        c("depth").mean(),
        c("base_depth").drop_nulls().first(),
        c("shelf_price").drop_nulls().first(),
        *[c(x).sum() for x in ["units", "cf_units", "incr_units", "net_rev", "cf_net_rev",
                               "cannibal_units", "cannibal_rev"]],
    ).sort(["product_id", "episode_id"])
    # Profit change at gross margin m (margin on the regular shelf price):
    #   unit cost c = (1 - m) * shelf price
    #   own change   = (net_rev - c * units) - (cf_net_rev - c * cf_units)
    #   cannibal     = m_net * cannibal_rev, using the same margin on the regular net price
    # Break even margin solves own change - cannibal = 0, which is linear in m.
    p = c("shelf_price")
    a = c("net_rev") - c("cf_net_rev") - p * (c("units") - c("cf_units"))
    slope = p * (c("units") - c("cf_units")) - c("cannibal_rev")
    return ep.with_columns(
        pl.when(slope > 0).then(-a / slope).alias("break_even_margin"),
        *[(a + m * slope).alias(f"profit_at_{int(m * 100)}") for m in MARGINS],
        pl.when(c("units") > 0).then(c("cf_units") / c("units")).alias("subsidized_share"),
        ((c("depth") - c("base_depth")).clip(lower_bound=0) * p * c("units")).alias("discount_dollars"),
    )


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    OUT.mkdir(exist_ok=True)
    panel = pl.read_parquet(PROC / "panel.parquet")
    df = prepare(panel)
    model = fit(df)
    eff = effects_table(model)
    coef = dict(zip(eff["term"], eff["Estimate"]))
    ep = deal_economics(df, coef)
    df.write_parquet(PROC / "panel_features.parquet")
    ep.write_parquet(PROC / "deal_episodes.parquet")
    eff.write_csv(OUT / "lift_effects.csv")
    ep.write_csv(OUT / "deal_episodes.csv")
    print(eff)
    print(ep.describe())


if __name__ == "__main__":
    main()
