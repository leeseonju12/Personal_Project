from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LapDeltaBandConfig:
    # Band thresholds in percent points
    band_a_max: float = 0.30
    band_b_max: float = 1.00

    # Monte Carlo settings
    n_sims: int = 10000
    seed: int = 42

    # Sigma model for rpi uncertainty
    # - "fixed": use fixed_sigma for all drivers (recommended baseline)
    # - "cs_linear": legacy linear map from cs_iqr_ms
    sigma_mode: str = "fixed"
    fixed_sigma: float = 0.12
    base_sigma: float = 0.08
    cs_scale: float = 0.00005
    sigma_floor: float = 0.08


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def simulate_lap_delta_bands(
    metrics_df: pd.DataFrame,
    cfg: LapDeltaBandConfig = LapDeltaBandConfig(),
    diagnostics: bool = False,
) -> pd.DataFrame:
    """
    Compute Lap Delta Band probabilities by sampling a future RPI distribution.

    Required columns:
      - driver_number
      - rpi_pct

    Optional columns:
      - cs_iqr_ms (used to scale uncertainty)
    """
    required = {"driver_number", "rpi_pct"}
    missing = required - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_df missing required columns: {sorted(missing)}")

    df = metrics_df.copy()

    # Ensure numeric
    df["rpi_pct"] = pd.to_numeric(df["rpi_pct"], errors="coerce")
    if "cs_iqr_ms" in df.columns:
        df["cs_iqr_ms"] = pd.to_numeric(df["cs_iqr_ms"], errors="coerce")

    # Drop drivers with no RPI
    df = df[df["rpi_pct"].notna()].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "driver_number",
                "band_a_prob",
                "band_b_prob",
                "band_c_prob",
                "mu_rpi",
                "sigma_rpi",
            ]
        )

    # Uncertainty per driver
    cs = df["cs_iqr_ms"] if "cs_iqr_ms" in df.columns else pd.Series(np.nan, index=df.index)
    cs = cs.fillna(cs.median() if cs.notna().any() else 0.0)
    if cfg.sigma_mode == "fixed":
        sigma_rpi = np.full(len(df), float(cfg.fixed_sigma), dtype=float)
    elif cfg.sigma_mode == "cs_linear":
        sigma_rpi = np.maximum(cfg.sigma_floor, cfg.base_sigma + cfg.cs_scale * cs.to_numpy(dtype=float))
    else:
        raise ValueError(f"Unsupported sigma_mode: {cfg.sigma_mode}")

    mu = df["rpi_pct"].to_numpy(dtype=float)
    drivers = pd.to_numeric(df["driver_number"], errors="coerce")

    rng = np.random.default_rng(cfg.seed)
    samples = rng.normal(loc=mu, scale=sigma_rpi, size=(cfg.n_sims, len(drivers)))

    # Band membership
    a = samples <= cfg.band_a_max
    b = (samples > cfg.band_a_max) & (samples <= cfg.band_b_max)
    c = samples > cfg.band_b_max

    out = pd.DataFrame(
        {
            "driver_number": drivers.astype("Int64").to_numpy(),
            "band_a_prob": a.mean(axis=0).astype(float),
            "band_b_prob": b.mean(axis=0).astype(float),
            "band_c_prob": c.mean(axis=0).astype(float),
            "mu_rpi": mu,
            "sigma_rpi": sigma_rpi,
        }
    )

    if diagnostics:
        z_a = (cfg.band_a_max - mu) / sigma_rpi
        z_b = (cfg.band_b_max - mu) / sigma_rpi
        p_a_closed = _normal_cdf(z_a)
        p_c_closed = 1.0 - _normal_cdf(z_b)
        out["band_a_closed"] = p_a_closed
        out["band_c_closed"] = p_c_closed
        out["band_a_abs_diff"] = np.abs(out["band_a_prob"] - out["band_a_closed"])
        out["band_c_abs_diff"] = np.abs(out["band_c_prob"] - out["band_c_closed"])

    return out.sort_values("band_a_prob", ascending=False).reset_index(drop=True)
