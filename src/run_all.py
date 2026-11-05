"""Build the panel, fit the model, cost every deal, run the household checks and write results.json."""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model  # noqa: E402
import halo  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"


def build_panel() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for t in ["transactions", "promotions", "products"]:
        con.sql(f"create view raw_{t} as select * from '{ROOT}/data/raw/{t}.parquet'")
    con.sql((ROOT / "sql" / "01_panel.sql").read_text())
    con.sql(f"copy panel to '{PROC}/panel.parquet'")
    con.sql(f"copy products_in_scope to '{PROC}/products_in_scope.parquet'")


def household_checks() -> dict:
    con = duckdb.connect()
    con.sql(f"create view tx as select * from '{ROOT}/data/raw/transactions.parquet'")
    con.sql(f"create view feat as select product_id, week, deal from '{PROC}/panel_features.parquet'")
    con.sql("create table hp as select t.household_id, t.product_id, t.week, max(f.deal) deal "
            "from tx t join feat f using (product_id, week) group by 1, 2, 3")
    d = con.sql("""
        select a.*,
          exists (select 1 from hp b where b.household_id = a.household_id and b.product_id = a.product_id
                  and b.week between a.week - 13 and a.week - 1) as seen,
          exists (select 1 from hp c where c.household_id = a.household_id and c.product_id = a.product_id
                  and c.week between a.week + 1 and a.week + 8 and c.deal = 0) as back_full,
          (select count(*) from feat f where f.product_id = a.product_id
                  and f.week between a.week + 1 and a.week + 8 and f.deal = 0) as full_weeks_available
        from hp a where a.week between 15 and 44
    """).pl().with_columns(pl.col(pl.Decimal).cast(pl.Float64))
    d = d.with_columns((~pl.col("seen")).cast(pl.Int64).alias("new"), pl.col("back_full").cast(pl.Int64))
    n = d.filter(pl.col("new") == 1)
    # pyfixest takes pandas frames, so convert only at the model boundary.
    new_fit = pf.feols("new ~ deal | product_id + week", data=d.to_pandas(), vcov={"CRV1": "product_id"}).tidy()
    back_fit = pf.feols("back_full ~ deal + full_weeks_available | product_id + week", data=n.to_pandas(),
                        vcov={"CRV1": "product_id"}).tidy()
    base_new = float(d.filter(pl.col("deal") == 0)["new"].mean())
    base_back = float(n.filter(pl.col("deal") == 0)["back_full"].mean())
    return {
        "purchases": d.height,
        "new_buyers_followed": n.height,
        "new_share_non_deal": base_new,
        "new_share_deal_effect_pts": float(new_fit.loc["deal", "Estimate"]) * 100,
        "new_share_deal_ci_pts": [float(new_fit.loc["deal", "2.5%"]) * 100, float(new_fit.loc["deal", "97.5%"]) * 100],
        "came_back_full_price_non_deal": base_back,
        "came_back_deal_effect_pts": float(back_fit.loc["deal", "Estimate"]) * 100,
        "came_back_deal_ci_pts": [float(back_fit.loc["deal", "2.5%"]) * 100, float(back_fit.loc["deal", "97.5%"]) * 100],
    }


def halo_block(ep: pl.DataFrame) -> dict:
    hw, regulars = halo.household_weeks()
    fit = halo.fit_halo(hw)
    per = fit["other_spend"]["per_deal"]
    lo, hi = fit["other_spend"]["ci"]
    return {
        "fit": fit,
        "adjusted": halo.halo_adjusted(ep, regulars, per),
        "adjusted_low": halo.halo_adjusted(ep, regulars, lo),
        "adjusted_high": halo.halo_adjusted(ep, regulars, hi),
    }


MARGIN_PCTS = [20, 25, 30, 35, 40]
DEPTH_BANDS = [(0.15, 0.2, "15 to 20%"), (0.2, 0.25, "20 to 25%"), (0.25, 0.3, "25 to 30%"),
               (0.3, 0.4, "30 to 40%"), (0.4, 1.0, "40% +")]


def summarize(ep: pl.DataFrame) -> dict:
    c = pl.col
    ep = ep.drop_nulls("shelf_price").with_columns(
        # A deal with no extra discount divides by zero. Treat that as missing, not as a number.
        (-c("profit_at_30") / c("discount_dollars")).clip(lower_bound=0).fill_nan(None).alias("funding_needed_30")
    )
    total_units = ep["units"].sum()
    out = {
        "deals": ep.height,
        "products": ep["product_id"].n_unique(),
        "deal_units": float(total_units),
        "subsidized_units": float(ep["cf_units"].sum()),
        "incremental_units": float(ep["incr_units"].sum()),
        "cannibalized_units": float(ep["cannibal_units"].sum()),
        "subsidized_share_weighted": float(ep["cf_units"].sum() / total_units),
        "subsidized_share_median": float(ep["subsidized_share"].median()),
        "break_even_margin_median": float(ep["break_even_margin"].median()),
        "funding_needed_30_median": float(ep["funding_needed_30"].median()),
        "discount_dollars": float(ep["discount_dollars"].sum()),
        "profitable_share_by_margin": {str(m): float((ep[f"profit_at_{m}"] > 0).mean()) for m in MARGIN_PCTS},
        "profit_by_margin": {str(m): float(ep[f"profit_at_{m}"].sum()) for m in MARGIN_PCTS},
    }

    def group(col):
        return (ep.drop_nulls(col).group_by(col).agg(
            pl.len().alias("deals"),
            c("break_even_margin").median().alias("break_even"),
            c("subsidized_share").median().alias("subsidized"),
            c("funding_needed_30").median().alias("funding"),
            (c("profit_at_30") > 0).mean().alias("profitable30"),
            *[(c(f"profit_at_{m}") > 0).mean().alias(f"profitable_{m}") for m in MARGIN_PCTS],
        ).with_columns(pl.col("deals").cast(pl.Int64)))

    band = pl.lit(None, dtype=pl.Utf8)
    for lo, hi, label in reversed(DEPTH_BANDS):
        band = pl.when((c("depth") > lo) & (c("depth") <= hi)).then(pl.lit(label)).otherwise(band)
    ep = ep.with_columns(
        band.alias("depth_band"),
        pl.when(c("mailer") == 1).then(pl.lit("In the weekly mailer")).otherwise(pl.lit("Shelf price cut only")).alias("advertised"),
    )
    order = {label: i for i, (_, _, label) in enumerate(DEPTH_BANDS)}
    by_depth = group("depth_band").sort(c("depth_band").replace_strict(order, return_dtype=pl.Int64))
    out["by_depth"] = by_depth.to_dicts()
    out["by_mailer"] = group("advertised").sort("advertised").to_dicts()
    dept = group("department").filter(c("deals") >= 50).sort(["deals", "department"], descending=[True, False])
    out["by_department"] = dept.to_dicts()
    return out


def sensitivity(panel: pl.DataFrame) -> list:
    rows = []
    for thr in [0.10, 0.15, 0.20]:
        model.DEAL_THRESHOLD = thr
        df = model.prepare(panel)
        fit = model.fit(df)
        coef = dict(zip(model.effects_table(fit)["term"], model.effects_table(fit)["Estimate"]))
        ep = model.deal_economics(df, coef).drop_nulls("shelf_price")
        rows.append({
            "threshold": thr,
            "elasticity": coef["log_price"],
            "mailer_effect_pct": (np.exp(coef["mailer_ad"]) - 1) * 100,
            "subsidized_share": float(ep["cf_units"].sum() / ep["units"].sum()),
            "profitable_share_30": float((ep["profit_at_30"] > 0).mean()),
            "break_even_median": float(ep["break_even_margin"].median()),
        })
    model.DEAL_THRESHOLD = 0.15
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    build_panel()
    panel = pl.read_parquet(PROC / "panel.parquet")
    df = model.prepare(panel)
    fit = model.fit(df)
    eff = model.effects_table(fit)
    coef = dict(zip(eff["term"], eff["Estimate"]))
    ep = model.deal_economics(df, coef)
    df.write_parquet(PROC / "panel_features.parquet")
    ep.write_parquet(PROC / "deal_episodes.parquet")
    eff.write_csv(OUT / "lift_effects.csv")
    ep.write_csv(OUT / "deal_episodes.csv")

    tx = duckdb.sql(f"select count(*) n, count(distinct household_id) hh, count(distinct basket_id) b "
                    f"from '{ROOT}/data/raw/transactions.parquet'").pl().row(0, named=True)
    results = {
        "data": {
            "transactions": int(tx["n"]), "households": int(tx["hh"]), "baskets": int(tx["b"]),
            "products_in_scope": df["product_id"].n_unique(), "weeks": "2 to 52 of 2017",
            "product_weeks": df.height, "deal_week_share": float(df["deal"].mean()),
            "deal_threshold": model.DEAL_THRESHOLD,
        },
        "effects": eff.to_dicts(),
        "deals": summarize(ep),
        "households": household_checks(),
        "halo": halo_block(ep),
        "sensitivity": sensitivity(panel),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: v for k, v in results.items() if k != "effects"}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
