from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import pandas as pd


def _to_df(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    raise TypeError(f"Unsupported payload type: {type(payload)!r}")


def _call_with_context(func: Callable[..., Any], context: Mapping[str, Any]) -> Any:
    sig = inspect.signature(func)
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in context:
            kwargs[name] = context[name]
        elif param.default is inspect.Parameter.empty:
            raise TypeError(f"Required argument '{name}' missing for {func.__name__}()")
    return func(**kwargs)


def _find_session_key(sessions: pd.DataFrame, keywords: tuple[str, ...]) -> Optional[int]:
    if sessions.empty:
        return None
    if "session_key" not in sessions.columns:
        return None

    type_col = sessions.get("session_type", pd.Series("", index=sessions.index)).astype(str)
    name_col = sessions.get("session_name", pd.Series("", index=sessions.index)).astype(str)
    text = (type_col + " " + name_col).str.lower()
    mask = pd.Series(False, index=sessions.index)
    for kw in keywords:
        mask = mask | text.str.contains(kw.lower(), na=False)

    matched = sessions.loc[mask, "session_key"]
    if matched.empty:
        return None
    return int(matched.iloc[0])


def _build_prediction_inputs(feat_metrics: pd.DataFrame, qpi_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["driver_number", "rpi_pct", "cs_iqr_ms", "pit_median_ms"]
    missing = [c for c in cols if c not in feat_metrics.columns]
    if missing:
        raise ValueError(f"feat_metrics is missing columns: {missing}")

    pred = feat_metrics[cols].copy()
    if "qpi_pct" in feat_metrics.columns:
        pred["qpi_pct"] = pd.to_numeric(feat_metrics["qpi_pct"], errors="coerce")
    elif not qpi_df.empty and {"driver_number", "qpi_pct"}.issubset(set(qpi_df.columns)):
        pred = pred.merge(
            qpi_df[["driver_number", "qpi_pct"]].copy(),
            how="left",
            on="driver_number",
        )
    else:
        pred["qpi_pct"] = pd.NA

    for c in ["rpi_pct", "cs_iqr_ms", "pit_median_ms", "qpi_pct"]:
        pred[c] = pd.to_numeric(pred[c], errors="coerce")
        pred[c] = pred[c].fillna(pred[c].median())
    # Final fallback for all-NaN columns.
    defaults = {"rpi_pct": 1.0, "cs_iqr_ms": 1500.0, "pit_median_ms": 25000.0, "qpi_pct": 1.0}
    for c, v in defaults.items():
        pred[c] = pred[c].fillna(v)
    return pred


def _default_save_outputs(
    output_dir: Path,
    meeting_key: int,
    race_session_key: int,
    band_probs: pd.DataFrame,
    finish_probs: pd.DataFrame,
) -> None:
    target_dir = output_dir / str(meeting_key)
    target_dir.mkdir(parents=True, exist_ok=True)
    band_probs.to_csv(target_dir / f"band_probs_session_{race_session_key}.csv", index=False)
    finish_probs.to_csv(target_dir / f"finish_probs_session_{race_session_key}.csv", index=False)


def run_one_meeting(
    meeting_key: int,
    openf1_get: Callable[[str, Mapping[str, Any]], Any],
    build_fact_race_result: Callable[..., pd.DataFrame],
    build_feat_metrics: Callable[..., pd.DataFrame],
    compute_qpi: Callable[..., pd.DataFrame],
    simulate_lap_delta_bands: Callable[..., pd.DataFrame],
    simulate_finish_group_probs: Callable[..., pd.DataFrame],
    save_outputs: Optional[Callable[..., None]] = None,
    output_dir: Optional[str] = None,
    min_driver_count: int = 20,
) -> dict[str, Any]:
    sessions = _to_df(openf1_get("sessions", {"meeting_key": meeting_key}))
    race_session_key = _find_session_key(sessions, ("race",))
    quali_session_key = _find_session_key(sessions, ("qualifying", "quali"))

    if race_session_key is None:
        return {"meeting_key": meeting_key, "status": "skipped", "reason": "race session not found"}

    race_result = _to_df(openf1_get("session_result", {"session_key": race_session_key}))
    race_laps = _to_df(openf1_get("laps", {"session_key": race_session_key}))
    race_pit = _to_df(openf1_get("pit", {"session_key": race_session_key}))
    quali_laps = (
        _to_df(openf1_get("laps", {"session_key": quali_session_key}))
        if quali_session_key is not None
        else pd.DataFrame()
    )

    context = {
        "meeting_key": meeting_key,
        "race_session_key": race_session_key,
        "quali_session_key": quali_session_key,
        "sessions": sessions,
        "race_result": race_result,
        "race_laps": race_laps,
        "race_pit": race_pit,
        "quali_laps": quali_laps,
    }

    fact_race = _call_with_context(build_fact_race_result, context)
    context["fact_race"] = fact_race
    feat_metrics = _call_with_context(build_feat_metrics, context)
    context["feat_metrics"] = feat_metrics
    qpi_df = _call_with_context(compute_qpi, context) if not quali_laps.empty else pd.DataFrame()

    metrics_pred = _build_prediction_inputs(feat_metrics, qpi_df)
    if len(metrics_pred) < min_driver_count:
        return {
            "meeting_key": meeting_key,
            "race_session_key": race_session_key,
            "quali_session_key": quali_session_key,
            "status": "skipped",
            "reason": f"driver count < {min_driver_count}",
        }

    band_probs = simulate_lap_delta_bands(metrics_pred)
    finish_probs = simulate_finish_group_probs(metrics_pred)

    if "finish_position" in fact_race.columns:
        labels = fact_race[["driver_number", "finish_position"]].copy()
        finish_pos = pd.to_numeric(labels["finish_position"], errors="coerce")
        labels["y_top10"] = (finish_pos <= 10).fillna(False).astype(int)
        labels["y_podium"] = (finish_pos <= 3).fillna(False).astype(int)
        finish_probs = finish_probs.merge(labels[["driver_number", "y_top10", "y_podium"]], how="left")

    if save_outputs is not None:
        save_outputs(
            meeting_key=meeting_key,
            race_session_key=race_session_key,
            sessions=sessions,
            fact_race=fact_race,
            feat_metrics=feat_metrics,
            qpi_df=qpi_df,
            race_pit=race_pit,
            band_probs=band_probs,
            finish_probs=finish_probs,
        )
    elif output_dir:
        _default_save_outputs(Path(output_dir), meeting_key, race_session_key, band_probs, finish_probs)

    return {
        "meeting_key": meeting_key,
        "race_session_key": race_session_key,
        "quali_session_key": quali_session_key,
        "fact_race": fact_race,
        "feat_metrics": feat_metrics,
        "qpi_df": qpi_df,
        "band_probs": band_probs,
        "finish_probs": finish_probs,
        "status": "ok",
    }


def run_season_pipeline(
    year: int,
    openf1_get: Callable[[str, Mapping[str, Any]], Any],
    build_fact_race_result: Callable[..., pd.DataFrame],
    build_feat_metrics: Callable[..., pd.DataFrame],
    compute_qpi: Callable[..., pd.DataFrame],
    simulate_lap_delta_bands: Callable[..., pd.DataFrame],
    simulate_finish_group_probs: Callable[..., pd.DataFrame],
    save_outputs: Optional[Callable[..., None]] = None,
    output_dir: Optional[str] = None,
    min_driver_count: int = 20,
) -> list[dict[str, Any]]:
    meetings = _to_df(openf1_get("meetings", {"year": year}))
    if meetings.empty or "meeting_key" not in meetings.columns:
        return []

    meetings = meetings.sort_values("meeting_key")
    outputs: list[dict[str, Any]] = []
    for meeting_key in meetings["meeting_key"].dropna().astype(int).tolist():
        result = run_one_meeting(
            meeting_key=meeting_key,
            openf1_get=openf1_get,
            build_fact_race_result=build_fact_race_result,
            build_feat_metrics=build_feat_metrics,
            compute_qpi=compute_qpi,
            simulate_lap_delta_bands=simulate_lap_delta_bands,
            simulate_finish_group_probs=simulate_finish_group_probs,
            save_outputs=save_outputs,
            output_dir=output_dir,
            min_driver_count=min_driver_count,
        )
        outputs.append(result)
    return outputs
