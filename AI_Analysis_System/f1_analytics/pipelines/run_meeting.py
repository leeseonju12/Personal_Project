from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from ..models_finish_group import simulate_finish_group_probs
from ..models_pace_band import simulate_lap_delta_bands
from .pipeline_season import run_one_meeting

OPENF1_BASE_URL = "https://api.openf1.org/v1"


@dataclass
class OpenF1Client:
    base_url: str = OPENF1_BASE_URL
    timeout_sec: int = 30

    def get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = requests.get(url, params=params, timeout=self.timeout_sec)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return [payload]
        except Exception as exc:
            print(f"[WARN] OpenF1 request failed: {path} params={params} err={exc}")
        return []


def _pick_meeting_by_country(client: OpenF1Client, year: int, country_name: str) -> int | None:
    meetings = pd.DataFrame(client.get("meetings", {"year": year}))
    if meetings.empty:
        return None

    text_cols = [
        c
        for c in ["country_name", "location", "meeting_name", "meeting_official_name", "circuit_short_name"]
        if c in meetings.columns
    ]
    if not text_cols:
        return None

    target = country_name.strip().lower()
    mask = pd.Series(False, index=meetings.index)
    for col in text_cols:
        mask = mask | meetings[col].astype(str).str.lower().str.contains(target, na=False)
    matched = meetings.loc[mask].sort_values("meeting_key")
    if matched.empty:
        return None
    return int(matched.iloc[-1]["meeting_key"])


def _build_fact_race_result(race_result: pd.DataFrame, race_session_key: int, **_: Any) -> pd.DataFrame:
    work = race_result.copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "driver_number",
                "team_name",
                "grid_position",
                "finish_position",
                "points",
                "status",
                "dnf",
                "dns",
                "dsq",
                "gap_to_leader",
                "number_of_laps",
                "duration",
            ]
        )

    if "position" in work.columns and "finish_position" not in work.columns:
        work["finish_position"] = pd.to_numeric(work["position"], errors="coerce")
    if "grid_position" not in work.columns:
        work["grid_position"] = pd.to_numeric(work.get("grid", np.nan), errors="coerce")

    def _series(col: str, default: Any = "") -> pd.Series:
        if col in work.columns:
            return work[col]
        return pd.Series([default] * len(work), index=work.index)

    status_text = _series("status", "").astype(str).str.lower()
    dnf = status_text.str.contains("dnf|ret|retired|not classified", regex=True, na=False).astype(int)
    dns = status_text.str.contains("dns|did not start", regex=True, na=False).astype(int)
    dsq = status_text.str.contains("dsq|disqual", regex=True, na=False).astype(int)

    out = pd.DataFrame(
        {
            "session_id": int(race_session_key),
            "driver_number": pd.to_numeric(work.get("driver_number"), errors="coerce").astype("Int64"),
            "team_name": work.get("team_name"),
            "grid_position": pd.to_numeric(work.get("grid_position"), errors="coerce").astype("Int64"),
            "finish_position": pd.to_numeric(work.get("finish_position"), errors="coerce").astype("Int64"),
            "points": pd.to_numeric(work.get("points"), errors="coerce"),
            "status": _series("status", "").astype(str),
            "dnf": dnf.astype(int),
            "dns": dns.astype(int),
            "dsq": dsq.astype(int),
            "gap_to_leader": _series("gap_to_leader", None),
            "number_of_laps": pd.to_numeric(work.get("number_of_laps"), errors="coerce").astype("Int64"),
            "duration": _series("duration", None),
        }
    )
    return out.dropna(subset=["driver_number"]).reset_index(drop=True)


def _prepare_laps(laps: pd.DataFrame) -> pd.DataFrame:
    work = laps.copy()
    if work.empty:
        return work

    # OpenF1 usually provides lap_duration in seconds for laps endpoint.
    if "lap_duration" in work.columns:
        work["lap_time_ms"] = pd.to_numeric(work["lap_duration"], errors="coerce") * 1000.0
    elif "lap_time" in work.columns:
        work["lap_time_ms"] = pd.to_timedelta(work["lap_time"], errors="coerce").dt.total_seconds() * 1000.0
    elif "lap_time_ms" in work.columns:
        work["lap_time_ms"] = pd.to_numeric(work["lap_time_ms"], errors="coerce")
    else:
        work["lap_time_ms"] = np.nan

    work["lap_number"] = pd.to_numeric(work.get("lap_number"), errors="coerce")
    work["driver_number"] = pd.to_numeric(work.get("driver_number"), errors="coerce")
    work["is_pit_out_lap"] = (
        work.get("is_pit_out_lap", pd.Series([0] * len(work), index=work.index))
        .astype(int)
        .fillna(0)
    )
    return work.dropna(subset=["driver_number", "lap_number", "lap_time_ms"]).query("lap_time_ms > 0")


def _prepare_pit(race_pit: pd.DataFrame) -> pd.DataFrame:
    work = race_pit.copy()
    if work.empty:
        return pd.DataFrame(columns=["driver_number", "stop_number", "pit_ms"])
    def _series(col: str, default: Any = np.nan) -> pd.Series:
        if col in work.columns:
            return work[col]
        return pd.Series([default] * len(work), index=work.index)

    work["driver_number"] = pd.to_numeric(work.get("driver_number"), errors="coerce")
    work["stop_number"] = pd.to_numeric(_series("stop_number", 1), errors="coerce").fillna(1).astype(int)
    if "pit_duration" in work.columns and "pit_ms" not in work.columns:
        work["pit_ms"] = pd.to_numeric(work["pit_duration"], errors="coerce") * 1000.0
    elif "stop_duration" in work.columns and "pit_ms" not in work.columns:
        work["pit_ms"] = pd.to_numeric(work["stop_duration"], errors="coerce") * 1000.0
    work["pit_ms"] = pd.to_numeric(work.get("pit_ms"), errors="coerce")
    return work.dropna(subset=["driver_number", "pit_ms"])


def _compute_qpi(quali_laps: pd.DataFrame, **_: Any) -> pd.DataFrame:
    laps = _prepare_laps(quali_laps)
    if laps.empty:
        return pd.DataFrame(columns=["driver_number", "qpi_pct"])

    best = laps.groupby("driver_number", as_index=False)["lap_time_ms"].min().rename(columns={"lap_time_ms": "best_ms"})
    ref = float(best["best_ms"].min())
    best["qpi_pct"] = ((best["best_ms"] - ref) / ref) * 100.0
    return best[["driver_number", "qpi_pct"]]


def _build_feat_metrics(
    race_laps: pd.DataFrame,
    race_pit: pd.DataFrame,
    race_session_key: int,
    fact_race: pd.DataFrame,
    **_: Any,
) -> pd.DataFrame:
    laps = _prepare_laps(race_laps)
    pits = _prepare_pit(race_pit)

    if laps.empty:
        base = fact_race[["driver_number"]].copy()
        for col in ["rpi_pct", "clean_lap_median_ms", "clean_lap_count", "cs_sd_ms", "cs_iqr_ms", "pit_median_ms", "pit_count", "sfd"]:
            base[col] = np.nan
        base["session_id"] = int(race_session_key)
        return base

    clean = laps[laps["is_pit_out_lap"] == 0].copy()
    if clean.empty:
        clean = laps.copy()

    grp = clean.groupby("driver_number")["lap_time_ms"]
    q1 = grp.quantile(0.25).rename("q1")
    q3 = grp.quantile(0.75).rename("q3")
    iqr = (q3 - q1).rename("cs_iqr_ms")
    lap_stats = pd.concat(
        [
            grp.median().rename("clean_lap_median_ms"),
            grp.count().rename("clean_lap_count"),
            grp.std(ddof=0).rename("cs_sd_ms"),
            iqr,
        ],
        axis=1,
    ).reset_index()

    ref = float(lap_stats["clean_lap_median_ms"].min())
    lap_stats["rpi_pct"] = ((lap_stats["clean_lap_median_ms"] - ref) / ref) * 100.0

    if pits.empty:
        pit_stats = pd.DataFrame({"driver_number": lap_stats["driver_number"], "pit_median_ms": np.nan, "pit_count": 0})
    else:
        pgrp = pits.groupby("driver_number")["pit_ms"]
        pit_stats = pd.concat([pgrp.median().rename("pit_median_ms"), pgrp.count().rename("pit_count")], axis=1).reset_index()

    out = lap_stats.merge(pit_stats, how="left", on="driver_number")
    out["pit_count"] = out["pit_count"].fillna(0).astype(int)
    out["pit_median_ms"] = out["pit_median_ms"].fillna(out["pit_median_ms"].median())

    if {"driver_number", "grid_position", "finish_position"}.issubset(fact_race.columns):
        sfd = fact_race[["driver_number", "grid_position", "finish_position"]].copy()
        sfd["sfd"] = pd.to_numeric(sfd["grid_position"], errors="coerce") - pd.to_numeric(
            sfd["finish_position"], errors="coerce"
        )
        out = out.merge(sfd[["driver_number", "sfd"]], how="left", on="driver_number")
    else:
        out["sfd"] = np.nan

    out["session_id"] = int(race_session_key)
    return out


def _save_outputs_to_db(engine: Engine, **kwargs: Any) -> None:
    fact_race: pd.DataFrame = kwargs["fact_race"]
    feat_metrics: pd.DataFrame = kwargs["feat_metrics"]
    qpi_df: pd.DataFrame = kwargs["qpi_df"]
    finish_probs: pd.DataFrame = kwargs["finish_probs"]
    race_session_key: int = int(kwargs["race_session_key"])

    if not qpi_df.empty and {"driver_number", "qpi_pct"}.issubset(feat_metrics.columns) is False:
        feat_metrics = feat_metrics.merge(qpi_df[["driver_number", "qpi_pct"]], how="left", on="driver_number")
    elif not qpi_df.empty and "qpi_pct" in feat_metrics.columns:
        q = qpi_df[["driver_number", "qpi_pct"]].copy()
        feat_metrics = feat_metrics.drop(columns=["qpi_pct"]).merge(q, how="left", on="driver_number")

    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM fact_race_result WHERE session_id = :sid"), {"sid": race_session_key})
            conn.execute(text("DELETE FROM feat_driver_session_metrics WHERE session_id = :sid"), {"sid": race_session_key})
            conn.execute(text("DELETE FROM forecast_race WHERE target_session_id = :sid"), {"sid": race_session_key})
    except Exception as exc:
        print(f"[WARN] pre-delete skipped: session={race_session_key} err={exc}")

    def _align_to_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        try:
            cols = [c["name"] for c in inspect(engine).get_columns(table_name)]
            if cols:
                return df[[c for c in df.columns if c in cols]].copy()
        except Exception:
            pass
        return df.copy()

    fact_race = _align_to_table(fact_race, "fact_race_result")
    feat_metrics = _align_to_table(feat_metrics, "feat_driver_session_metrics")
    fact_race.to_sql("fact_race_result", engine, if_exists="append", index=False)
    feat_metrics.to_sql("feat_driver_session_metrics", engine, if_exists="append", index=False)

    forecast = finish_probs.copy()
    forecast["target_session_id"] = race_session_key
    forecast["exp_finish_pos"] = forecast["exp_rank"]
    forecast = forecast[
        ["target_session_id", "driver_number", "exp_points", "podium_prob", "top10_prob", "dnf_prob", "exp_finish_pos"]
    ]
    forecast = _align_to_table(forecast, "forecast_race")
    forecast.to_sql("forecast_race", engine, if_exists="append", index=False)


def run_meeting_by_country(engine: Engine, year: int, country_name: str) -> list[dict[str, Any]]:
    client = OpenF1Client()
    meeting_key = _pick_meeting_by_country(client, year=year, country_name=country_name)
    if meeting_key is None:
        print(f"[WARN] no meeting found for year={year}, country={country_name}")
        return [{"forecast": pd.DataFrame(columns=["driver_number", "exp_points", "top10_prob", "podium_prob"])}]

    out = run_one_meeting(
        meeting_key=meeting_key,
        openf1_get=client.get,
        build_fact_race_result=_build_fact_race_result,
        build_feat_metrics=_build_feat_metrics,
        compute_qpi=_compute_qpi,
        simulate_lap_delta_bands=simulate_lap_delta_bands,
        simulate_finish_group_probs=simulate_finish_group_probs,
        save_outputs=lambda **kwargs: _save_outputs_to_db(engine=engine, **kwargs),
        min_driver_count=10,
    )

    if out.get("status") != "ok":
        print(f"[WARN] meeting skipped: {out}")
        return [{"forecast": pd.DataFrame(columns=["driver_number", "exp_points", "top10_prob", "podium_prob"])}]

    forecast = out["finish_probs"][["driver_number", "exp_points", "top10_prob", "podium_prob"]].copy()
    return [{"forecast": forecast, **out}]
