from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None


def _to_ms(series: pd.Series) -> pd.Series:
    # If already numeric (OpenF1 lap_duration is usually seconds), infer by magnitude.
    s = pd.to_numeric(series, errors="coerce")
    # Values in seconds are typically < 1000, milliseconds are much larger.
    return np.where(s < 1000, s * 1000.0, s)


def normalize_quali_laps(laps_df: pd.DataFrame) -> pd.DataFrame:
    df = laps_df.copy()
    if df.empty:
        return df

    drv_col = _pick_col(df, ["driver_number", "driver_no", "driver"])
    lap_no_col = _pick_col(df, ["lap_number", "lap", "lap_no"])
    lap_ms_col = _pick_col(df, ["lap_time_ms"])
    lap_dur_col = _pick_col(df, ["lap_duration", "lap_time"])

    if lap_ms_col:
        df["lap_time_ms"] = pd.to_numeric(df[lap_ms_col], errors="coerce")
    elif lap_dur_col:
        if str(df[lap_dur_col].dtype) == "object":
            df["lap_time_ms"] = pd.to_timedelta(df[lap_dur_col], errors="coerce").dt.total_seconds() * 1000.0
        else:
            df["lap_time_ms"] = _to_ms(df[lap_dur_col]).astype(float)
    else:
        df["lap_time_ms"] = np.nan

    if drv_col:
        df["driver_number"] = pd.to_numeric(df[drv_col], errors="coerce")
    else:
        df["driver_number"] = np.nan
    if lap_no_col:
        df["lap_number"] = pd.to_numeric(df[lap_no_col], errors="coerce")
    else:
        df["lap_number"] = np.nan

    invalid_mask = pd.Series(False, index=df.index)
    del_col = _pick_col(df, ["is_deleted", "deleted"])
    valid_col = _pick_col(df, ["lap_valid", "is_valid"])
    if del_col:
        invalid_mask = invalid_mask | df[del_col].astype(bool)
    if valid_col:
        invalid_mask = invalid_mask | (~df[valid_col].astype(bool))

    keep = df["driver_number"].notna() & df["lap_time_ms"].notna() & (df["lap_time_ms"] > 0) & (~invalid_mask)
    df = df[keep]

    date_col = _pick_col(df, ["date_start", "lap_start_time", "date"])
    if date_col:
        df["_sort_dt"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(["driver_number", "_sort_dt", "lap_number"], na_position="last")
        df = df.drop(columns=["_sort_dt"])
    else:
        df = df.sort_values(["driver_number", "lap_number"], na_position="last")
    return df.reset_index(drop=True)


def compute_best_lap_qpi(laps_df: pd.DataFrame) -> pd.DataFrame:
    laps = normalize_quali_laps(laps_df)
    if laps.empty:
        return pd.DataFrame(columns=["driver_number", "best_lap_ms", "qpi_pct"])

    out = laps.groupby("driver_number", as_index=False)["lap_time_ms"].min().rename(columns={"lap_time_ms": "best_lap_ms"})
    session_best = float(out["best_lap_ms"].min())
    out["qpi_pct"] = (out["best_lap_ms"] / session_best - 1.0) * 100.0
    return out


def compute_teammate_delta(best_qpi_df: pd.DataFrame, session_result_df: pd.DataFrame) -> pd.DataFrame:
    qpi = best_qpi_df.copy()
    if qpi.empty:
        return pd.DataFrame(columns=["driver_number", "teammate_delta"])

    sr = session_result_df.copy()
    team_col = _pick_col(sr, ["team_name", "team", "constructor_name"])
    drv_col = _pick_col(sr, ["driver_number", "driver_no", "driver"])
    if team_col is None or drv_col is None:
        out = qpi[["driver_number"]].copy()
        out["teammate_delta"] = np.nan
        return out

    sr["driver_number"] = pd.to_numeric(sr[drv_col], errors="coerce")
    sr = sr[["driver_number", team_col]].rename(columns={team_col: "team_name"})
    merged = qpi.merge(sr, on="driver_number", how="left")
    team_counts = merged.groupby("team_name")["driver_number"].transform("count")
    team_best = merged.groupby("team_name")["qpi_pct"].transform("min")
    merged["teammate_delta"] = np.where(team_counts >= 2, merged["qpi_pct"] - team_best, np.nan)
    return merged[["driver_number", "teammate_delta"]]


def compute_sector_pcts(laps_df: pd.DataFrame) -> pd.DataFrame:
    laps = normalize_quali_laps(laps_df)
    if laps.empty:
        return pd.DataFrame(columns=["driver_number", "s1_pct", "s2_pct", "s3_pct"])

    s1_col = _pick_col(laps, ["duration_sector_1", "sector_1_time"])
    s2_col = _pick_col(laps, ["duration_sector_2", "sector_2_time"])
    s3_col = _pick_col(laps, ["duration_sector_3", "sector_3_time"])
    if not (s1_col and s2_col and s3_col):
        return pd.DataFrame(columns=["driver_number", "s1_pct", "s2_pct", "s3_pct"])

    work = laps[["driver_number", s1_col, s2_col, s3_col]].copy()
    work[s1_col] = pd.to_numeric(work[s1_col], errors="coerce")
    work[s2_col] = pd.to_numeric(work[s2_col], errors="coerce")
    work[s3_col] = pd.to_numeric(work[s3_col], errors="coerce")

    best = work.groupby("driver_number", as_index=False).min(numeric_only=True)
    sb1 = float(best[s1_col].min())
    sb2 = float(best[s2_col].min())
    sb3 = float(best[s3_col].min())
    out = pd.DataFrame(
        {
            "driver_number": best["driver_number"],
            "s1_pct": (best[s1_col] / sb1 - 1.0) * 100.0,
            "s2_pct": (best[s2_col] / sb2 - 1.0) * 100.0,
            "s3_pct": (best[s3_col] / sb3 - 1.0) * 100.0,
        }
    )
    return out


def compute_quali_progress(laps_df: pd.DataFrame, session_result_df: pd.DataFrame) -> pd.DataFrame:
    laps = normalize_quali_laps(laps_df)
    sr = session_result_df.copy()

    drivers_from_laps = pd.Series(laps["driver_number"].unique(), name="driver_number") if not laps.empty else pd.Series([], name="driver_number")
    drv_col = _pick_col(sr, ["driver_number", "driver_no", "driver"])
    drivers_from_sr = pd.to_numeric(sr[drv_col], errors="coerce").dropna().unique() if drv_col else np.array([])
    drivers = np.unique(np.concatenate([drivers_from_laps.to_numpy(dtype=float), drivers_from_sr.astype(float)]))
    out = pd.DataFrame({"driver_number": drivers})
    if out.empty:
        return pd.DataFrame(columns=["driver_number", "in_q1", "in_q2", "in_q3"])

    out["in_q1"] = False
    out["in_q2"] = False
    out["in_q3"] = False

    seg_col = _pick_col(laps, ["segment", "session_part", "quali_segment", "part"])
    if seg_col and not laps.empty:
        seg = laps[["driver_number", seg_col]].copy()
        seg[seg_col] = seg[seg_col].astype(str).str.upper()
        grouped = seg.groupby("driver_number")[seg_col].agg(lambda x: set(x.dropna()))
        out = out.set_index("driver_number")
        out["in_q1"] = out.index.isin(grouped.index)
        out["in_q2"] = [("Q2" in grouped.get(i, set())) for i in out.index]
        out["in_q3"] = [("Q3" in grouped.get(i, set())) for i in out.index]
        return out.reset_index()

    q1_col = _pick_col(sr, ["q1_time", "q1", "q1_lap_time"])
    q2_col = _pick_col(sr, ["q2_time", "q2", "q2_lap_time"])
    q3_col = _pick_col(sr, ["q3_time", "q3", "q3_lap_time"])
    if drv_col and (q1_col or q2_col or q3_col):
        tmp = sr.copy()
        tmp["driver_number"] = pd.to_numeric(tmp[drv_col], errors="coerce")
        tmp = tmp.dropna(subset=["driver_number"])
        out = out.set_index("driver_number")
        base = tmp.set_index("driver_number")
        out["in_q1"] = base[q1_col].notna() if q1_col else True
        out["in_q2"] = base[q2_col].notna() if q2_col else False
        out["in_q3"] = base[q3_col].notna() if q3_col else False
        out = out.fillna(False).reset_index()
        return out

    out["in_q1"] = True if not laps.empty else False
    return out


def compute_improvement(laps_df: pd.DataFrame) -> pd.DataFrame:
    laps = normalize_quali_laps(laps_df)
    if laps.empty:
        return pd.DataFrame(columns=["driver_number", "improvement_ms", "improvement_pct"])

    first = laps.groupby("driver_number", as_index=False).first()[["driver_number", "lap_time_ms"]].rename(
        columns={"lap_time_ms": "first_valid_lap_ms"}
    )
    best = laps.groupby("driver_number", as_index=False)["lap_time_ms"].min().rename(columns={"lap_time_ms": "best_lap_ms"})
    out = first.merge(best, on="driver_number", how="inner")
    out["improvement_ms"] = out["first_valid_lap_ms"] - out["best_lap_ms"]
    out["improvement_pct"] = np.where(
        out["first_valid_lap_ms"] > 0,
        out["improvement_ms"] / out["first_valid_lap_ms"] * 100.0,
        np.nan,
    )
    return out[["driver_number", "improvement_ms", "improvement_pct"]]


def estimate_sigma_q_from_laps(
    quali_laps: pd.DataFrame,
    best_lap_df: pd.DataFrame,
    *,
    k_fast: int = 5,
    min_laps: int = 3,
    min_sigma: float = 0.06,
    max_sigma: float = 0.30,
    fallback_sigma: float = 0.12,
) -> pd.Series:
    """
    Estimate per-driver sigma for QPI% using top-k fastest laps' IQR.
    Returns Series indexed by driver_number.
    """
    laps = normalize_quali_laps(quali_laps)
    if "lap_time_ms" not in laps.columns:
        raise ValueError("quali_laps must have lap_time_ms")

    best_map = (
        best_lap_df[["driver_number", "best_lap_ms"]]
        .dropna()
        .assign(driver_number=lambda x: pd.to_numeric(x["driver_number"], errors="coerce"))
        .dropna(subset=["driver_number"])
        .set_index("driver_number")["best_lap_ms"]
        .astype(float)
    )

    sigmas: dict[int, float] = {}
    for drv, part in laps.groupby("driver_number"):
        x = part["lap_time_ms"].dropna().astype(float).to_numpy()
        drv_i = int(drv)
        if len(x) < min_laps or drv not in best_map.index:
            sigmas[drv_i] = np.nan
            continue

        x = np.sort(x)
        topk = x[: min(k_fast, len(x))]
        if len(topk) < min_laps:
            sigmas[drv_i] = np.nan
            continue

        q25, q75 = np.percentile(topk, [25, 75])
        iqr = float(q75 - q25)
        sigma_ms = iqr / 1.349 if iqr > 0 else 0.0
        best_ms = float(best_map.loc[drv])
        sigma_pct = (sigma_ms / best_ms) * 100.0 if best_ms > 0 else np.nan
        sigmas[drv_i] = sigma_pct

    sigma_s = pd.Series(sigmas, dtype="float64")
    global_sigma = sigma_s.dropna().median() if sigma_s.notna().any() else fallback_sigma
    sigma_s = sigma_s.fillna(global_sigma).clip(lower=min_sigma, upper=max_sigma)
    return sigma_s


def simulate_quali_probs(
    qpi_df: pd.DataFrame,
    quali_laps: pd.DataFrame | None = None,
    sigma_source: str = "per_driver",
    n_sims: int = 5000,
    fixed_sigma: float = 0.12,
    seed: int = 42,
) -> pd.DataFrame:
    required = {"driver_number", "qpi_pct"}
    missing = required - set(qpi_df.columns)
    if missing:
        raise ValueError(f"qpi_df missing required columns: {sorted(missing)}")

    df = qpi_df.copy()
    df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce")
    df["qpi_pct"] = pd.to_numeric(df["qpi_pct"], errors="coerce")
    df = df.dropna(subset=["driver_number", "qpi_pct"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["driver_number", "pole_prob", "top3_prob", "top10_prob", "exp_grid_pos", "sigma_q"])

    if sigma_source == "per_driver":
        if quali_laps is not None and {"driver_number", "best_lap_ms"}.issubset(df.columns):
            sigma_s = estimate_sigma_q_from_laps(
                quali_laps=quali_laps[["driver_number", "lap_time_ms"]] if "lap_time_ms" in quali_laps.columns else quali_laps,
                best_lap_df=df[["driver_number", "best_lap_ms"]],
                k_fast=5,
                min_laps=3,
                min_sigma=0.06,
                max_sigma=0.30,
                fallback_sigma=fixed_sigma,
            )
            sigma = sigma_s.reindex(df["driver_number"].astype(int)).reset_index(drop=True)
        else:
            sigma_col = _pick_col(df, ["sigma_q", "qpi_sigma", "sigma"])
            if sigma_col:
                sigma = pd.to_numeric(df[sigma_col], errors="coerce")
                sigma = sigma.fillna(sigma.median() if sigma.notna().any() else fixed_sigma).clip(lower=0.06, upper=0.30)
            else:
                sigma = pd.Series(fixed_sigma, index=df.index)
    elif sigma_source == "fixed":
        sigma = pd.Series(fixed_sigma, index=df.index)
    else:
        raise ValueError("sigma_source must be 'per_driver' or 'fixed'")

    mu = df["qpi_pct"].to_numpy(dtype=float)
    sig = sigma.to_numpy(dtype=float)
    n = len(df)
    rng = np.random.default_rng(seed)
    sims = rng.normal(loc=mu, scale=sig, size=(n_sims, n))

    pole_hits = np.zeros(n, dtype=float)
    top3_hits = np.zeros(n, dtype=float)
    top10_hits = np.zeros(n, dtype=float)
    rank_sum = np.zeros(n, dtype=float)

    for i in range(n_sims):
        order = np.argsort(sims[i])  # lower qpi is better
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        pole_hits += (ranks == 1).astype(float)
        top3_hits += (ranks <= 3).astype(float)
        top10_hits += (ranks <= 10).astype(float)
        rank_sum += ranks

    out = pd.DataFrame(
        {
            "driver_number": df["driver_number"].astype("Int64"),
            "pole_prob": pole_hits / n_sims,
            "top3_prob": top3_hits / n_sims,
            "top10_prob": top10_hits / n_sims,
            "exp_grid_pos": rank_sum / n_sims,
            "sigma_q": sig,
        }
    )
    return out.sort_values("pole_prob", ascending=False).reset_index(drop=True)


def compute_qpi_range(driver_history_df: pd.DataFrame, n: int = 5, std_floor: float = 0.08) -> pd.DataFrame:
    required = {"driver_number", "qpi_pct"}
    missing = required - set(driver_history_df.columns)
    if missing:
        raise ValueError(f"driver_history_df missing required columns: {sorted(missing)}")

    df = driver_history_df.copy()
    df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce")
    df["qpi_pct"] = pd.to_numeric(df["qpi_pct"], errors="coerce")
    df = df.dropna(subset=["driver_number", "qpi_pct"])
    if df.empty:
        return pd.DataFrame(columns=["driver_number", "qpi_range_low", "qpi_range_high"])

    sort_col = _pick_col(df, ["session_id", "session_key", "date", "created_at"])
    if sort_col:
        if "date" in sort_col.lower() or "created" in sort_col.lower():
            df[sort_col] = pd.to_datetime(df[sort_col], errors="coerce")
        df = df.sort_values(["driver_number", sort_col])

    rows: list[dict[str, float]] = []
    for driver, g in df.groupby("driver_number"):
        tail = g["qpi_pct"].tail(n)
        mu = float(tail.mean())
        sd = float(tail.std(ddof=0)) if len(tail) > 1 else 0.0
        sd = max(sd, std_floor)
        rows.append({"driver_number": int(driver), "qpi_range_low": mu - sd, "qpi_range_high": mu + sd})
    return pd.DataFrame(rows)
