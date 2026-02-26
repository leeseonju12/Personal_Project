from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def brier_score(prob: pd.Series, label: pd.Series) -> float:
    p = pd.to_numeric(prob, errors="coerce").clip(0, 1)
    y = pd.to_numeric(label, errors="coerce")
    mask = p.notna() & y.notna()
    if not mask.any():
        return float("nan")
    return float(np.mean((p[mask] - y[mask]) ** 2))


def calibration_table(
    prob: pd.Series,
    label: pd.Series,
    bins: Iterable[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.00001),
) -> pd.DataFrame:
    p = pd.to_numeric(prob, errors="coerce").clip(0, 1)
    y = pd.to_numeric(label, errors="coerce")
    mask = p.notna() & y.notna()
    if not mask.any():
        return pd.DataFrame(columns=["bin", "n", "pred_mean", "actual_rate"])

    work = pd.DataFrame({"p": p[mask], "y": y[mask]})
    work["bin"] = pd.cut(work["p"], bins=list(bins), right=False, include_lowest=True)

    agg = (
        work.groupby("bin", observed=True)
        .agg(n=("p", "size"), pred_mean=("p", "mean"), actual_rate=("y", "mean"))
        .reset_index()
    )
    agg["bin"] = agg["bin"].astype(str)
    return agg


def evaluate_finish_group_forecast(
    finish_probs: pd.DataFrame,
    fact_race: pd.DataFrame,
    pred_session_col: str = "target_session_id",
    fact_session_col: str = "session_id",
) -> dict[str, object]:
    pred = finish_probs.copy()
    fact = fact_race.copy()

    if pred_session_col not in pred.columns:
        pred[pred_session_col] = np.nan
    if fact_session_col not in fact.columns:
        fact[fact_session_col] = np.nan

    key_pred = [pred_session_col, "driver_number"] if pred_session_col in pred.columns else ["driver_number"]
    key_fact = [fact_session_col, "driver_number"] if fact_session_col in fact.columns else ["driver_number"]

    cols = key_fact + ["finish_position"]
    merged = pred.merge(
        fact[cols].copy(),
        how="left",
        left_on=key_pred,
        right_on=key_fact,
    )

    merged["y_top10"] = (pd.to_numeric(merged["finish_position"], errors="coerce") <= 10).astype(float)
    merged["y_podium"] = (pd.to_numeric(merged["finish_position"], errors="coerce") <= 3).astype(float)

    result = {
        "brier_top10": brier_score(merged["top10_prob"], merged["y_top10"]),
        "brier_podium": brier_score(merged["podium_prob"], merged["y_podium"]),
        "calibration_top10": calibration_table(merged["top10_prob"], merged["y_top10"]),
        "calibration_podium": calibration_table(merged["podium_prob"], merged["y_podium"]),
        "top10_pick_accuracy": top_k_pick_accuracy(merged, "top10_prob", "finish_position", k=10),
        "podium_pick_accuracy": top_k_pick_accuracy(merged, "podium_prob", "finish_position", k=3),
        "merged": merged,
    }
    return result


def top_k_pick_accuracy(
    merged_pred_fact: pd.DataFrame,
    prob_col: str,
    finish_col: str = "finish_position",
    k: int = 10,
    session_col: str = "target_session_id",
) -> float:
    work = merged_pred_fact.copy()
    if session_col not in work.columns:
        picked = work.nlargest(k, prob_col)
        return float((pd.to_numeric(picked[finish_col], errors="coerce") <= k).mean())

    acc_list: list[float] = []
    for _, g in work.groupby(session_col, dropna=True):
        top = g.nlargest(k, prob_col)
        y = (pd.to_numeric(top[finish_col], errors="coerce") <= k).mean()
        if pd.notna(y):
            acc_list.append(float(y))
    return float(np.mean(acc_list)) if acc_list else float("nan")
