"""Write docs/data.js for the static dashboard from outputs/results.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    r = json.loads((ROOT / "outputs" / "results.json").read_text())
    eff = {e["term"]: e for e in r["effects"]}
    elasticity = eff["log_price"]["Estimate"]
    mailer = eff["mailer_ad"]["Estimate"]
    curve = []
    for d in range(0, 55, 5):
        base = (1 - d / 100) ** elasticity
        curve.append({"depth": d, "shelf_only": base, "with_mailer": base * (2.718281828 ** mailer)})
    timeline = [
        {"label": label, "pct": eff[t]["effect_pct"], "lo": eff[t]["ci_low_pct"], "hi": eff[t]["ci_high_pct"]}
        for label, t in [("2 wks before", "pre_2"), ("1 wk before", "pre_1"), ("1 wk after", "post_1"),
                         ("2 wks after", "post_2"), ("3 wks after", "post_3"), ("4 wks after", "post_4")]
    ]
    weeks = [
        {"label": label, "pct": eff[t]["effect_pct"], "lo": eff[t]["ci_low_pct"], "hi": eff[t]["ci_high_pct"]}
        for label, t in [("Week 2 of deal", "deal_w2"), ("Week 3+ of deal", "deal_w3plus")]
    ]
    data = {
        "data": r["data"],
        "deals": r["deals"],
        "households": r["households"],
        "halo": r["halo"],
        "sensitivity": r["sensitivity"],
        "effects": {k: {kk: v[kk] for kk in ["effect_pct", "ci_low_pct", "ci_high_pct", "Estimate"]} for k, v in eff.items()},
        "curve": curve,
        "timeline": timeline,
        "deal_weeks": weeks,
    }
    out = ROOT / "docs" / "data.js"
    out.write_text("window.PROMO = " + json.dumps(data, default=float) + ";\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
