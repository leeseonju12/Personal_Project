import os

import pandas as pd

from f1_analytics.db import get_engine
from f1_analytics.models_strategy_analysis import build_strategy_analysis
from f1_analytics.pipelines.run_meeting import run_meeting_by_country

TARGET_YEAR = int(os.getenv("F1_YEAR", "2025"))
TARGET_COUNTRY = os.getenv("F1_COUNTRY", "Mexico")


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
        safe = safe.sort_values(sort_by, ascending=True if sort_by in {"qpi_pct", "exp_grid_pos"} else False)
    safe = safe.head(head_n).reset_index(drop=True)
    print(safe.to_string(index=False))
    print("=" * (len(title) + 12))


def main() -> None:
    engine = get_engine()
    outputs = run_meeting_by_country(engine, year=TARGET_YEAR, country_name=TARGET_COUNTRY)
    print(f"[DONE] run finished: country={TARGET_COUNTRY}, sessions={len(outputs)}")
    if outputs:
        first = outputs[0]
        if "quali_summary" in first and not first["quali_summary"].empty:
            _print_df_section(
                "Qualifying Summary Top10",
                first["quali_summary"],
                ["driver_number", "best_lap_ms", "qpi_pct", "teammate_delta", "grid_position"],
                sort_by="qpi_pct",
            )

        if "quali_forecast" in first and not first["quali_forecast"].empty:
            _print_df_section(
                "Qualifying Forecast Top10",
                first["quali_forecast"],
                ["driver_number", "pole_prob", "top3_prob", "top10_prob", "exp_grid_pos", "sigma_q"],
                sort_by="exp_grid_pos",
            )

        _print_df_section(
            "Forecast Top10",
            first.get("forecast", pd.DataFrame()),
            ["driver_number", "exp_points", "top10_prob", "podium_prob"],
            sort_by="exp_points",
        )

        if "band_probs" in first:
            _print_df_section(
                "Lap Delta Band Top10",
                first["band_probs"],
                ["driver_number", "band_a_prob", "band_b_prob", "band_c_prob", "mu_rpi", "sigma_rpi"],
                sort_by="band_a_prob",
            )

        race_session_key = first.get("race_session_key")
        if race_session_key is not None:
            strategy_df = first.get("strategy_analysis", pd.DataFrame())
            if strategy_df is None or strategy_df.empty:
                strategy_df = build_strategy_analysis(engine, session_id=int(race_session_key))
            _print_df_section(
                "Strategy Analysis Top10",
                strategy_df,
                [
                    "driver_number",
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
