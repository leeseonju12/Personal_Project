from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

FIA_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)


def _zscore(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).all():
        arr = np.zeros_like(arr)
    else:
        arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
    mean = arr.mean()
    std = arr.std(ddof=0)
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _compute_score(work: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    z_rpi = _zscore(work["rpi_pct"])
    z_cs = _zscore(work["cs_iqr_ms"])
    z_pit = _zscore(work["pit_median_ms"])

    if "qpi_pct" in work.columns and not work["qpi_pct"].isna().all():
        z_qpi = _zscore(work["qpi_pct"])
        weights = np.array([0.55, 0.20, 0.10, 0.15], dtype=float)
        score = -(weights[0] * z_rpi + weights[1] * z_cs + weights[2] * z_pit + weights[3] * z_qpi)
    else:
        weights = np.array([0.55, 0.20, 0.10], dtype=float)
        weights = weights / weights.sum()
        score = -(weights[0] * z_rpi + weights[1] * z_cs + weights[2] * z_pit)

    return score, z_cs


def simulate_finish_group_probs(
    metrics: pd.DataFrame,
    n_sim: int = 3000,
    random_state: Optional[int] = None,
    debug_print: bool = True,
) -> pd.DataFrame:
    required = {"driver_number", "rpi_pct", "cs_iqr_ms", "pit_median_ms"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Explicitly keep prediction-safe columns only (leakage guard).
    allow_cols = ["driver_number", "rpi_pct", "cs_iqr_ms", "pit_median_ms", "qpi_pct"]
    work = metrics[[c for c in allow_cols if c in metrics.columns]].copy()

    for col in ["rpi_pct", "cs_iqr_ms", "pit_median_ms", "qpi_pct"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            work[col] = work[col].fillna(work[col].median())

    score, z_cs = _compute_score(work)
    sigma = 0.35 * (1.0 + 0.35 * np.clip(z_cs, 0, None))
    p_dnf = np.clip(0.03 + 0.03 * np.clip(z_cs, 0, None), 0.01, 0.35)

    n_drivers = len(work)
    if debug_print:
        print(f"[DBG] N={n_drivers}")
        print(f"[DBG] rpi_pct min/max={work['rpi_pct'].min():.6f}/{work['rpi_pct'].max():.6f}")
        print(f"[DBG] cs_iqr_ms min/max={work['cs_iqr_ms'].min():.6f}/{work['cs_iqr_ms'].max():.6f}")
        print(f"[DBG] pit_median_ms min/max={work['pit_median_ms'].min():.6f}/{work['pit_median_ms'].max():.6f}")
        print(f"[DBG] strength min/max/std={score.min():.6f}/{score.max():.6f}/{score.std(ddof=0):.6f}")
        print(f"[DBG] sim_sigma_mean={sigma.mean():.6f}")

    rng = np.random.default_rng(random_state)
    perf = rng.normal(loc=score, scale=sigma, size=(n_sim, n_drivers))
    dnf_mask = rng.random(size=(n_sim, n_drivers)) < p_dnf

    top10_hits = np.zeros(n_drivers, dtype=float)
    podium_hits = np.zeros(n_drivers, dtype=float)
    dnf_hits = np.zeros(n_drivers, dtype=float)
    rank_sum = np.zeros(n_drivers, dtype=float)
    points_sum = np.zeros(n_drivers, dtype=float)

    for i in range(n_sim):
        alive = np.where(~dnf_mask[i])[0]
        dnf = np.where(dnf_mask[i])[0]
        dnf_hits[dnf] += 1

        order_alive = alive[np.argsort(-perf[i, alive])] if alive.size else np.array([], dtype=int)
        order_dnf = dnf[rng.permutation(len(dnf))] if dnf.size else np.array([], dtype=int)
        final_order = np.concatenate([order_alive, order_dnf])

        ranks = np.empty(n_drivers, dtype=int)
        ranks[final_order] = np.arange(1, n_drivers + 1)

        top10_hits += (ranks <= 10).astype(float)
        podium_hits += (ranks <= 3).astype(float)
        rank_sum += ranks

        top_n = min(10, n_drivers)
        pos_to_driver = final_order[:top_n]
        points_sum[pos_to_driver] += FIA_POINTS[:top_n]

    return pd.DataFrame(
        {
            "driver_number": work["driver_number"].to_numpy(),
            "top10_prob": top10_hits / n_sim,
            "podium_prob": podium_hits / n_sim,
            "dnf_prob": dnf_hits / n_sim,
            "exp_rank": rank_sum / n_sim,
            "exp_points": points_sum / n_sim,
            "score_mean": score,
            "score_sigma": sigma,
        }
    )
