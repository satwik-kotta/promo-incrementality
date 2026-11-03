"""Build the panel, fit the model, cost every deal, run the household checks and write results.json."""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model  # noqa: E402

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
    """).df()
    d["new"] = (~d["seen"]).astype(int)
    d["back_full"] = d["back_full"].astype(int)
    new_fit = pf.feols("new ~ deal | product_id + week", data=d, vcov={"CRV1": "product_id"}).tidy()
    n = d[d["new"] == 1]
    back_fit = pf.feols("back_full ~ deal + full_weeks_available | product_id + week", data=n,
                        vcov={"CRV1": "product_id"}).tidy()
    base_new = float(d.loc[d["deal"] == 0, "new"].mean())
    base_back = float(n.loc[n["deal"] == 0, "back_full"].mean())
    return {
        "purchases": int(len(d)),
        "new_buyers_followed": int(len(n)),
        "new_share_non_deal": base_new,
        "new_share_deal_effect_pts": float(new_fit.loc["deal", "Estimate"]) * 100,
        "new_share_deal_ci_pts": [float(new_fit.loc["deal", "2.5%"]) * 100, float(new_fit.loc["deal", "97.5%"]) * 100],
        "came_back_full_price_non_deal": base_back,
        "came_back_deal_effect_pts": float(back_fit.loc["deal", "Estimate"]) * 100,
        "came_back_deal_ci_pts": [float(back_fit.loc["deal", "2.5%"]) * 100, float(back_fit.loc["deal", "97.5%"]) * 100],
    }


def summarize(ep: pd.DataFrame) -> dict:
    ep = ep.dropna(subset=["shelf_price"]).copy()
    ep["funding_needed_30"] = (-ep["profit_at_30"] / ep["discount_dollars"]).clip(lower=0)
    total_units = ep["units"].sum()
    out = {
        "deals": int(len(ep)),
        "products": int(ep["product_id"].nunique()),
        "deal_units": float(total_units),
        "subsidized_units": float(ep["cf_units"].sum()),
        "incremental_units": float(ep["incr_units"].sum()),
        "cannibalized_units": float(ep["cannibal_units"].sum()),
        "subsidized_share_weighted": float(ep["cf_units"].sum() / total_units),
        "subsidized_share_median": float(ep["subsidized_share"].median()),
        "break_even_margin_median": float(ep["break_even_margin"].median()),
        "funding_needed_30_median": float(ep["funding_needed_30"].median()),
        "discount_dollars": float(ep["discount_dollars"].sum()),
        "profitable_share_by_margin": {str(m): float((ep[f"profit_at_{m}"] > 0).mean()) for m in [20, 25, 30, 35, 40]},
        "profit_by_margin": {str(m): float(ep[f"profit_at_{m}"].sum()) for m in [20, 25, 30, 35, 40]},
    }

    def group(col):
        g = ep.groupby(col, observed=True).agg(
            deals=("units", "size"),
            break_even=("break_even_margin", "median"),
            subsidized=("subsidized_share", "median"),
            funding=("funding_needed_30", "median"),
            profitable30=("profit_at_30", lambda s: float((s > 0).mean())),
            profitable_by=("profit_at_20", "size"),
        ).reset_index()
        for m in [20, 25, 30, 35, 40]:
            g[f"profitable_{m}"] = ep.groupby(col, observed=True)[f"profit_at_{m}"].apply(lambda s: float((s > 0).mean())).values
        return g.drop(columns=["profitable_by"])

    ep["depth_band"] = pd.cut(ep["depth"], [0.15, 0.2, 0.25, 0.3, 0.4, 1.0],
                              labels=["15 to 20%", "20 to 25%", "25 to 30%", "30 to 40%", "40% +"])
    ep["advertised"] = np.where(ep["mailer"] == 1, "In the weekly mailer", "Shelf price cut only")
    out["by_depth"] = group("depth_band").to_dict(orient="records")
    out["by_mailer"] = group("advertised").to_dict(orient="records")
    dept = group("department")
    out["by_department"] = dept[dept["deals"] >= 50].sort_values("deals", ascending=False).to_dict(orient="records")
    return out


def sensitivity(panel: pd.DataFrame) -> list:
    rows = []
    for thr in [0.10, 0.15, 0.20]:
        model.DEAL_THRESHOLD = thr
        df = model.prepare(panel)
        fit = model.fit(df)
        coef = dict(zip(model.effects_table(fit)["term"], model.effects_table(fit)["Estimate"]))
        ep = model.deal_economics(df, coef).dropna(subset=["shelf_price"])
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
    panel = pd.read_parquet(PROC / "panel.parquet")
    df = model.prepare(panel)
    fit = model.fit(df)
    eff = model.effects_table(fit)
    coef = dict(zip(eff["term"], eff["Estimate"]))
    ep = model.deal_economics(df, coef)
    df.to_parquet(PROC / "panel_features.parquet")
    ep.to_parquet(PROC / "deal_episodes.parquet")
    eff.to_csv(OUT / "lift_effects.csv", index=False)
    ep.to_csv(OUT / "deal_episodes.csv", index=False)

    tx = duckdb.sql(f"select count(*) n, count(distinct household_id) hh, count(distinct basket_id) b "
                    f"from '{ROOT}/data/raw/transactions.parquet'").df().iloc[0]
    results = {
        "data": {
            "transactions": int(tx["n"]), "households": int(tx["hh"]), "baskets": int(tx["b"]),
            "products_in_scope": int(df["product_id"].nunique()), "weeks": "2 to 52 of 2017",
            "product_weeks": int(len(df)), "deal_week_share": float(df["deal"].mean()),
            "deal_threshold": model.DEAL_THRESHOLD,
        },
        "effects": eff.to_dict(orient="records"),
        "deals": summarize(ep),
        "households": household_checks(),
        "sensitivity": sensitivity(panel),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: v for k, v in results.items() if k != "effects"}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
