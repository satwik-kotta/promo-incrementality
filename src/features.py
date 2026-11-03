"""Event time features for each product week.

A promotion episode is a run of consecutive weeks in the weekly mailer. For every product week
this module records where it sits relative to the nearest episode, so one model can estimate
the lift in each promotion week, the dip after it ends, and a placebo check before it starts.
"""
import numpy as np
import pandas as pd

POST_WEEKS = 4
PRE_WEEKS = 2


def add_episode_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["product_id", "week"]).reset_index(drop=True).copy()
    df["episode_id"] = -1
    df["promo_week_no"] = 0
    df["episode_length"] = 0
    df["weeks_after"] = 0
    df["weeks_before"] = 0

    out = []
    for pid, g in df.groupby("product_id", sort=False):
        g = g.copy()
        m = g["mailer"].to_numpy()
        n = len(m)
        ep = np.full(n, -1)
        pos = np.zeros(n, dtype=int)
        length = np.zeros(n, dtype=int)
        starts, ends = [], []
        i = 0
        k = 0
        while i < n:
            if m[i] == 1:
                j = i
                while j + 1 < n and m[j + 1] == 1:
                    j += 1
                ep[i:j + 1] = k
                pos[i:j + 1] = np.arange(1, j - i + 2)
                length[i:j + 1] = j - i + 1
                starts.append(i)
                ends.append(j)
                k += 1
                i = j + 1
            else:
                i += 1
        after = np.zeros(n, dtype=int)
        before = np.zeros(n, dtype=int)
        after_ep = np.full(n, -1)
        for e, end in enumerate(ends):
            for h in range(1, POST_WEEKS + 1):
                t = end + h
                if t >= n or m[t] == 1:
                    break  # the after window ends when the next deal starts
                after[t] = h
                after_ep[t] = e
        for start in starts:
            for h in range(1, PRE_WEEKS + 1):
                t = start - h
                if t >= 0 and m[t] == 0 and after[t] == 0 and before[t] == 0:
                    before[t] = h
        g["episode_id"] = ep
        g["promo_week_no"] = pos
        g["episode_length"] = length
        g["weeks_after"] = after
        g["after_episode_id"] = after_ep
        g["weeks_before"] = before
        out.append(g)
    df = pd.concat(out, ignore_index=True)

    # Indicators used by the model. The reference group is a week with no promotion nearby.
    df["promo_w1"] = (df["promo_week_no"] == 1).astype(int)
    df["promo_w2"] = (df["promo_week_no"] == 2).astype(int)
    df["promo_w3plus"] = (df["promo_week_no"] >= 3).astype(int)
    for h in range(1, POST_WEEKS + 1):
        df[f"post_{h}"] = (df["weeks_after"] == h).astype(int)
    for h in range(1, PRE_WEEKS + 1):
        df[f"pre_{h}"] = (df["weeks_before"] == h).astype(int)
    df["display"] = (df["display_share"] > 0).astype(int)
    df["dept_week"] = df["department"].astype(str) + "_" + df["week"].astype(str)
    return df


def add_sibling_promo(df: pd.DataFrame) -> pd.DataFrame:
    """Share of other products in the same category that are in the mailer that week."""
    g = df.groupby(["product_category", "week"])["mailer"]
    total = g.transform("sum")
    count = g.transform("count")
    others = (count - 1).clip(lower=1)
    df = df.copy()
    df["sibling_promo_share"] = (total - df["mailer"]) / others
    df["n_in_category"] = count
    return df
