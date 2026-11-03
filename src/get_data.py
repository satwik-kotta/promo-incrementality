"""Download the Complete Journey data (84.51°, a Kroger company) and save it as Parquet.

Source: the open source `completejourney` R package by Brad Boehmke,
https://github.com/bradleyboehmke/completejourney (data originated from the 84.51° Complete Journey study).
"""
from pathlib import Path
import urllib.request

import pandas as pd
import pyreadr

BASE = "https://raw.githubusercontent.com/bradleyboehmke/completejourney/master/data/"
FILES = {
    "transactions": "transactions.rds",
    "promotions": "promotions.rds",
    "products": "products.rda",
    "campaigns": "campaigns.rda",
    "campaign_descriptions": "campaign_descriptions.rda",
    "coupons": "coupons.rda",
    "coupon_redemptions": "coupon_redemptions.rda",
    "demographics": "demographics.rda",
}
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, fname in FILES.items():
        out = RAW / f"{name}.parquet"
        if out.exists():
            print(f"have {out.name}")
            continue
        src = RAW / fname
        print(f"downloading {fname}")
        urllib.request.urlretrieve(BASE + fname, src)
        df = next(iter(pyreadr.read_r(str(src)).values()))
        for col in df.columns:
            if str(df[col].dtype).startswith("string"):
                df[col] = df[col].astype(object)
        df.to_parquet(out)
        src.unlink()
        print(f"wrote {out.name} {df.shape}")


if __name__ == "__main__":
    main()
