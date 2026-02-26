from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def simulate_lap_delta_bands(
    metrics: pd.DataFrame,
    n_sim: int = 2000,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    required = {"driver_number", "rpi_pct"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = metrics.copy()
    if "cs_iqr_ms" not in work.columns:
        work["cs_iqr_ms"] = np.nan

    work["rpi_pct"] = pd.to_numeric(work["rpi_pct"], errors="coerce")
    work["cs_iqr_ms"] = pd.to_numeric(work["cs_iqr_ms"], errors="coerce")

    work["rpi_pct"] = work["rpi_pct"].fillna(work["rpi_pct"].median())
    work["cs_iqr_ms"] = work["cs_iqr_ms"].fillna(work["cs_iqr_ms"].median())

    mu = work["rpi_pct"].to_numpy(dtype=float)
    sigma = np.maximum(0.08, 0.08 + 0.00005 * work["cs_iqr_ms"].to_numpy(dtype=float))

    rng = np.random.default_rng(random_state)
    samples = rng.normal(loc=mu, scale=sigma, size=(n_sim, len(work)))

    band_a = (samples <= 0.30).mean(axis=0)
    band_b = ((samples > 0.30) & (samples <= 1.00)).mean(axis=0)
    band_c = (samples > 1.00).mean(axis=0)

    return pd.DataFrame(
        {
            "driver_number": work["driver_number"].to_numpy(),
            "band_a_prob": band_a,
            "band_b_prob": band_b,
            "band_c_prob": band_c,
            "mu_rpi": mu,
            "sigma_rpi": sigma,
        }
    )

