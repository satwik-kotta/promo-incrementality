"""Basket halo: do deals bring shoppers in and lift spending on everything else?

Each household has a set of regular products (in scope products it bought in 3 or more weeks).
Every week a different number of those products happen to be on deal, and that number differs
across households because their regular baskets differ. Comparing the same household across weeks,
with week fixed effects absorbing chain wide seasonality and promotions, the effect of one more
deal on a household's regular products on its trips and on its spending on items that are not on
deal is the halo. Next week's deals are included as a placebo: they should have no effect today.
"""
from pathlib import Path

import duckdb
import polars as pl
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MARGINS = [20, 25, 30, 35, 40]


def household_weeks() -> tuple[pl.DataFrame, pl.DataFrame]:
    con = duckdb.connect()
    con.sql(f"create view tx as select * from '{ROOT}/data/raw/transactions.parquet' where week between 2 and 52")
    con.sql(f"create view deal as select product_id, week, deal from '{PROC}/panel_features.parquet'")
    con.sql("""
        create table reg as
        select household_id, product_id from tx
        where product_id in (select distinct product_id from deal)
        group by 1, 2 having count(distinct week) >= 3
    """)
    con.sql("""
        create table expo as
        select r.household_id, d.week, sum(d.deal) as n_deal, count(*) as n_regular
        from reg r join deal d using (product_id) group by 1, 2
    """)
    con.sql("""
        create table hw as
        select t.household_id, t.week,
               count(distinct t.basket_id) as trips,
               sum(t.sales_value) as spend,
               sum(case when coalesce(d.deal, 0) = 1 then t.sales_value else 0 end) as deal_spend,
               sum(case when coalesce(d.deal, 0) = 0 then t.sales_value else 0 end) as other_spend
        from tx t left join deal d using (product_id, week) group by 1, 2
    """)
    df = con.sql("""
        select e.household_id, e.week, e.n_deal, e.n_regular,
               lead(e.n_deal) over (partition by e.household_id order by e.week) as n_deal_next,
               coalesce(h.trips, 0) as trips, coalesce(h.spend, 0) as spend,
               coalesce(h.deal_spend, 0) as deal_spend, coalesce(h.other_spend, 0) as other_spend
        from expo e left join hw h using (household_id, week)
    """).pl().with_columns(pl.col(pl.Decimal).cast(pl.Float64))  # DuckDB sums arrive as decimals
    regulars = con.sql("select product_id, count(*) as regular_households from reg group by 1").pl()
    return df, regulars


def fit_halo(df: pl.DataFrame) -> dict:
    out = {}
    d = df.drop_nulls("n_deal_next")
    dp = d.to_pandas()  # pyfixest takes a pandas frame
    for y in ["trips", "other_spend", "deal_spend", "spend"]:
        t = pf.feols(f"{y} ~ n_deal + n_deal_next | household_id + week", data=dp,
                     vcov={"CRV1": "household_id"}).tidy()
        out[y] = {
            "per_deal": float(t.loc["n_deal", "Estimate"]),
            "ci": [float(t.loc["n_deal", "2.5%"]), float(t.loc["n_deal", "97.5%"])],
            "placebo_next_week": float(t.loc["n_deal_next", "Estimate"]),
            "placebo_ci": [float(t.loc["n_deal_next", "2.5%"]), float(t.loc["n_deal_next", "97.5%"])],
            "mean": float(d[y].mean()),
        }
    out["household_weeks"] = d.height
    out["households"] = d["household_id"].n_unique()
    return out


def halo_adjusted(episodes: pl.DataFrame, regulars: pl.DataFrame, other_spend_per_deal: float) -> dict:
    """Add the margin on halo spending to each deal, per week the deal runs."""
    ep = (episodes.drop_nulls("shelf_price")
          .join(regulars, on="product_id", how="left", maintain_order="left")
          .with_columns(pl.col("regular_households").fill_null(0)))
    ep = ep.with_columns((other_spend_per_deal * pl.col("regular_households") * pl.col("weeks")).alias("halo_spend"))
    res = {"halo_spend_total": float(ep["halo_spend"].sum())}
    for m in MARGINS:
        adj = ep[f"profit_at_{m}"] + (m / 100) * ep["halo_spend"]
        res[f"profitable_{m}"] = float((adj > 0).mean())
        res[f"profit_{m}"] = float(adj.sum())
    return res


if __name__ == "__main__":
    df, regulars = household_weeks()
    h = fit_halo(df)
    print(h)
    ep = pl.read_parquet(PROC / "deal_episodes.parquet")
    print(halo_adjusted(ep, regulars, h["other_spend"]["per_deal"]))
