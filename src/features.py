"""Event time features for each product week, in polars.

A promotion episode is a run of consecutive deal weeks. For every product week this module records
where it sits relative to the nearest episode, so one model can estimate the lift in each deal week,
the dip after it ends, and a placebo check before it starts.
"""
import numpy as np
import polars as pl

POST_WEEKS = 4
PRE_WEEKS = 2


def _episodes(m: np.ndarray) -> dict:
    """Episode arrays for one product's weekly 0/1 deal flags, in week order."""
    n = len(m)
    ep = np.full(n, -1)
    pos = np.zeros(n, dtype=np.int64)
    length = np.zeros(n, dtype=np.int64)
    starts, ends = [], []
    i = k = 0
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
    after = np.zeros(n, dtype=np.int64)
    before = np.zeros(n, dtype=np.int64)
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
    return {"episode_id": ep, "promo_week_no": pos, "episode_length": length,
            "weeks_after": after, "after_episode_id": after_ep, "weeks_before": before}


def add_episode_features(panel: pl.DataFrame) -> pl.DataFrame:
    df = panel.sort(["product_id", "week"])
    m = df["mailer"].to_numpy()
    ids = df["product_id"].to_numpy()
    n = len(df)
    cols = {k: np.zeros(n, dtype=np.int64) for k in
            ["episode_id", "promo_week_no", "episode_length", "weeks_after", "after_episode_id", "weeks_before"]}
    cuts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, n]):
        for k, v in _episodes(m[a:b]).items():
            cols[k][a:b] = v
    df = df.with_columns([pl.Series(k, v) for k, v in cols.items()])

    # Indicators used by the model. The reference group is a week with no promotion nearby.
    flag = lambda e: e.cast(pl.Int64)  # noqa: E731
    return df.with_columns(
        flag(pl.col("promo_week_no") == 1).alias("promo_w1"),
        flag(pl.col("promo_week_no") == 2).alias("promo_w2"),
        flag(pl.col("promo_week_no") >= 3).alias("promo_w3plus"),
        *[flag(pl.col("weeks_after") == h).alias(f"post_{h}") for h in range(1, POST_WEEKS + 1)],
        *[flag(pl.col("weeks_before") == h).alias(f"pre_{h}") for h in range(1, PRE_WEEKS + 1)],
        flag(pl.col("display_share") > 0).alias("display"),
        pl.concat_str([pl.col("department").cast(pl.Utf8), pl.col("week").cast(pl.Utf8)], separator="_").alias("dept_week"),
    )


def add_sibling_promo(df: pl.DataFrame) -> pl.DataFrame:
    """Share of other products in the same category that are on deal that week."""
    by = ["product_category", "week"]
    total = pl.col("mailer").sum().over(by)
    count = pl.col("mailer").count().over(by)
    return df.with_columns(
        ((total - pl.col("mailer")) / (count - 1).clip(lower_bound=1)).alias("sibling_promo_share"),
        count.alias("n_in_category"),
    )
