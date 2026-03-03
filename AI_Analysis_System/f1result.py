import os
import argparse

import pandas as pd
import requests

from f1_analytics.db import get_engine
from f1_analytics.models_strategy_analysis import build_strategy_analysis
from f1_analytics.pipelines.run_meeting import run_meeting_by_country

TARGET_YEAR = int(os.getenv("F1_YEAR", "2025"))
TARGET_COUNTRY = os.getenv("F1_COUNTRY", "Mexico")
OPENF1_BASE_URL = "https://api.openf1.org/v1"


def _attach_driver_meta(df: pd.DataFrame, fact_race: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    if fact_race is None or fact_race.empty:
        return df
    if "driver_number" not in df.columns or "driver_number" not in fact_race.columns:
        return df

    ref_cols = [
        c
        for c in ["driver_number", "driver_name", "team_name", "starting_grid_position", "finish_position"]
        if c in fact_race.columns
    ]
    if "driver_number" not in ref_cols:
        return df

    ref = fact_race[ref_cols].drop_duplicates("driver_number", keep="last").copy()
    merged = df.merge(ref, on="driver_number", how="left")
    return merged
 

def _print_df_section(title: str, df: pd.DataFrame, columns: list[str], sort_by: str | None = None, head_n: int = 10) -> None:
    print(f"\n===== {title} =====")
    if df is None or df.empty:
        print("(empty)")
        return

    safe = df.copy()
    keep = [c for c in columns if c in safe.columns]
    if not keep:
        print("(no expected columns)")
        return

    safe = safe[keep].copy()
    if sort_by and sort_by in safe.columns:
        safe = safe.sort_values(
            sort_by,
            ascending=True if sort_by in {"qpi_pct", "exp_grid_pos", "starting_grid_position"} else False,
        )
    safe = safe.head(head_n).reset_index(drop=True)
    print(safe.to_string(index=False))
    print("=" * (len(title) + 12))


def _openf1_get(path: str, params: dict[str, int]) -> pd.DataFrame:
    url = f"{OPENF1_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()
        payload = res.json()
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
    except Exception as exc:
        print(f"[WARN] OpenF1 request failed: path={path}, params={params}, err={exc}")
    return pd.DataFrame()


def _print_starting_grid(meeting_key: int | None, session_key: int) -> None:
    sessions = _openf1_get("sessions", {"session_key": session_key})
    if sessions.empty:
        print(f"[WARN] session not found: session_key={session_key}")
        return

    if meeting_key is not None and "meeting_key" in sessions.columns:
        session_meeting = pd.to_numeric(sessions["meeting_key"], errors="coerce").dropna()
        if not session_meeting.empty and int(session_meeting.iloc[0]) != int(meeting_key):
            print(
                f"[WARN] meeting_key mismatch: requested={meeting_key}, "
                f"session_meeting_key={int(session_meeting.iloc[0])}"
            )
            return

    starting_grid = _openf1_get("starting_grid", {"session_key": session_key})
    if starting_grid.empty and meeting_key is not None:
        starting_grid = _openf1_get("starting_grid", {"meeting_key": meeting_key})
    session_result = _openf1_get("session_result", {"session_key": session_key})
    drivers = _openf1_get("drivers", {"session_key": session_key})
    if starting_grid.empty and session_result.empty:
        print(f"[WARN] no starting_grid/session_result for session_key={session_key}")
        return

    grid = starting_grid.copy() if not starting_grid.empty else session_result.copy()
    if "starting_grid_position" in grid.columns:
        grid["starting_grid_position"] = pd.to_numeric(grid["starting_grid_position"], errors="coerce")
    elif "position" in grid.columns:
        grid["starting_grid_position"] = pd.to_numeric(grid["position"], errors="coerce")
    elif "grid_position" in grid.columns:
        grid["starting_grid_position"] = pd.to_numeric(grid["grid_position"], errors="coerce")
    else:
        grid["starting_grid_position"] = pd.to_numeric(grid.get("grid"), errors="coerce")

    if "driver_number" not in grid.columns:
        print(f"[WARN] driver_number missing in session_result: session_key={session_key}")
        return
    grid["driver_number"] = pd.to_numeric(grid["driver_number"], errors="coerce")
    grid = grid.dropna(subset=["driver_number"]).copy()

    if not drivers.empty and "driver_number" in drivers.columns:
        meta = drivers.copy()
        meta["driver_number"] = pd.to_numeric(meta["driver_number"], errors="coerce")
        for c in ["full_name", "team_name", "name_acronym"]:
            if c not in meta.columns:
                meta[c] = pd.NA
        meta = meta[["driver_number", "full_name", "name_acronym", "team_name"]].drop_duplicates("driver_number", keep="last")
        grid = grid.merge(meta, on="driver_number", how="left")

    grid = grid.sort_values(["starting_grid_position", "driver_number"], na_position="last").reset_index(drop=True)
    title = f"Starting Grid (meeting_key={meeting_key}, session_key={session_key})"
    _print_df_section(
        title,
        grid,
        ["starting_grid_position", "driver_number", "name_acronym", "full_name", "team_name"],
        sort_by="starting_grid_position",
        head_n=40,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 analytics runner / starting grid viewer")
    parser.add_argument("--meeting-key", type=int, default=None, help="OpenF1 meeting_key")
    parser.add_argument("--session-key", type=int, default=None, help="OpenF1 session_key")
    parser.add_argument("--starting-grid", action="store_true", help="Print starting grid for the given session")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.starting_grid or args.session_key is not None:
        if args.session_key is None:
            print("[WARN] --session-key is required for starting-grid mode")
            return
        _print_starting_grid(meeting_key=args.meeting_key, session_key=args.session_key)
        return

    engine = get_engine()
    outputs = run_meeting_by_country(engine, year=TARGET_YEAR, country_name=TARGET_COUNTRY)
    print(f"[DONE] run finished: country={TARGET_COUNTRY}, sessions={len(outputs)}")
    if outputs:
        first = outputs[0]
        fact_race = first.get("fact_race", pd.DataFrame())
        if "quali_summary" in first and not first["quali_summary"].empty:
            quali_summary = _attach_driver_meta(first["quali_summary"], fact_race)
            _print_df_section(
                "Qualifying Summary Top10",
                quali_summary,
                [
                    "driver_number",
                    "driver_name",
                    "team_name",
                    "starting_grid_position",
                    "best_lap_ms",
                    "qpi_pct",
                    "teammate_delta",
                ],
                sort_by="qpi_pct",
            )

        if "quali_forecast" in first and not first["quali_forecast"].empty:
            quali_forecast = _attach_driver_meta(first["quali_forecast"], fact_race)
            _print_df_section(
                "Qualifying Forecast Top10",
                quali_forecast,
                [
                    "driver_number",
                    "driver_name",
                    "team_name",
                    "starting_grid_position",
                    "exp_grid_pos",
                    "pole_prob",
                    "top3_prob",
                    "top10_prob",
                    "sigma_q",
                ],
                sort_by="exp_grid_pos",
            )

        forecast_df = _attach_driver_meta(first.get("forecast", pd.DataFrame()), fact_race)
        _print_df_section(
            "Forecast Top10",
            forecast_df,
            [
                "driver_number",
                "driver_name",
                "team_name",
                "finish_position",
                "exp_finish_pos",
                "exp_points",
                "top10_prob",
                "podium_prob",
            ],
            sort_by="exp_points",
        )

        if "band_probs" in first:
            band_probs = _attach_driver_meta(first["band_probs"], fact_race)
            _print_df_section(
                "Lap Delta Band Top10",
                band_probs,
                [
                    "driver_number",
                    "driver_name",
                    "team_name",
                    "finish_position",
                    "band_a_prob",
                    "band_b_prob",
                    "band_c_prob",
                    "mu_rpi",
                    "sigma_rpi",
                ],
                sort_by="band_a_prob",
            )

        race_session_key = first.get("race_session_key")
        if race_session_key is not None:
            strategy_df = first.get("strategy_analysis", pd.DataFrame())
            if strategy_df is None or strategy_df.empty:
                strategy_df = build_strategy_analysis(engine, session_id=int(race_session_key))
            strategy_df = _attach_driver_meta(strategy_df, fact_race)
            _print_df_section(
                "Strategy Analysis Top10",
                strategy_df,
                [
                    "driver_number",
                    "driver_name",
                    "team_name",
                    "finish_position",
                    "pit_count",
                    "pit_median_ms",
                    "strategy_type",
                    "pit_loss_percentile",
                    "ir_pct",
                    "n_ir",
                ],
                sort_by="pit_loss_percentile",
            )

if __name__ == "__main__":
    main()
