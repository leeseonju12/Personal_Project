from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class StrategyAnalysisConfig:
    enable_ir: bool = False
    ir_n: int = 5
    include_dns_in_ir: bool = False
    percentile_higher_is_worse: bool = True
    short_race_lap_threshold: int = 35
    save_table: str | None = None
    debug: bool = True
    raise_on_read_error: bool = False


def _read_df(
    engine: Engine,
    sql: str,
    params: dict,
    *,
    name: str = "query",
    debug: bool = True,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as exc:
        if debug:
            print(f"[ERR] {name} failed: {exc}")
            print(f"[ERR] sql={sql.strip()} params={params}")
        if raise_on_error:
            raise
        return pd.DataFrame()


def compute_pe_from_pit(pit_df: pd.DataFrame, drivers_df: pd.DataFrame) -> pd.DataFrame:
    out = drivers_df[["driver_number"]].drop_duplicates().copy()
    out["driver_number"] = pd.to_numeric(out["driver_number"], errors="coerce")
    if pit_df.empty:
        out["pit_count"] = 0
        out["pit_median_ms"] = np.nan
        return out

    work = pit_df.copy()
    work["driver_number"] = pd.to_numeric(work["driver_number"], errors="coerce")
    work["pit_ms"] = pd.to_numeric(work["pit_ms"], errors="coerce")
    work = work.dropna(subset=["driver_number", "pit_ms"])

    pe = (
        work.groupby("driver_number", as_index=False)
        .agg(pit_count=("pit_ms", "size"), pit_median_ms=("pit_ms", "median"))
        .copy()
    )
    out = out.merge(pe, on="driver_number", how="left")
    out["pit_count"] = out["pit_count"].fillna(0).astype(int)
    return out


def classify_strategy_type(pe_df: pd.DataFrame, short_race: bool = False) -> pd.DataFrame:
    out = pe_df.copy()
    if short_race:
        out["strategy_type"] = "ShortRace"
        return out

    out["strategy_type"] = np.select(
        [
            out["pit_count"] == 0,
            out["pit_count"] == 1,
            out["pit_count"] == 2,
            out["pit_count"] >= 3,
        ],
        [
            "No-stop/Unknown",
            "1-stop",
            "2-stop",
            "3-stop+",
        ],
        default="Unknown",
    )
    return out


def compute_pit_loss_percentile(pe_df: pd.DataFrame, higher_is_worse: bool = True) -> pd.DataFrame:
    out = pe_df.copy()
    out["pit_loss_percentile"] = np.nan

    mask = out["pit_median_ms"].notna()
    valid = out.loc[mask, ["driver_number", "pit_median_ms"]].copy()
    n = len(valid)
    if n == 0:
        return out
    if n == 1:
        out.loc[mask, "pit_loss_percentile"] = 0.0
        return out

    valid["rank"] = valid["pit_median_ms"].rank(method="min", ascending=True)
    pct = 100.0 * (valid["rank"] - 1.0) / (n - 1.0)
    valid["pit_loss_percentile"] = pct if higher_is_worse else 100.0 - pct
    out = out.drop(columns=["pit_loss_percentile"]).merge(
        valid[["driver_number", "pit_loss_percentile"]], on="driver_number", how="left"
    )
    return out


def compute_ir_pct_by_date(
    engine: Engine,
    current_session_id: int,
    ir_n: int = 5,
    include_dns: bool = False,
    *,
    debug: bool = True,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    cols = ["driver_number", "ir_pct", "n_ir"]
    sql = """
    SELECT
      r.session_id,
      r.driver_number,
      r.dnf, r.dns, r.dsq,
      s.date_start
    FROM fact_race_result r
    JOIN dim_session s ON s.session_id = r.session_id
    WHERE s.date_start < (SELECT date_start FROM dim_session WHERE session_id = :sid)
      AND (s.session_type IS NULL OR LOWER(s.session_type) LIKE '%race%')
    ORDER BY s.date_start DESC
    """
    h = _read_df(
        engine,
        sql,
        {"sid": current_session_id},
        name="ir_history_by_date",
        debug=debug,
        raise_on_error=raise_on_error,
    )
    if h.empty:
        return pd.DataFrame(columns=cols)

    h["driver_number"] = pd.to_numeric(h["driver_number"], errors="coerce")
    h["dnf"] = pd.to_numeric(h["dnf"], errors="coerce").fillna(0).astype(int)
    h["dsq"] = pd.to_numeric(h["dsq"], errors="coerce").fillna(0).astype(int)
    h["dns"] = pd.to_numeric(h["dns"], errors="coerce").fillna(0).astype(int)
    h = h.dropna(subset=["driver_number"])

    base_incident = ((h["dnf"] == 1) | (h["dsq"] == 1)).astype(int)
    if include_dns:
        base_incident = ((base_incident == 1) | (h["dns"] == 1)).astype(int)
    h["incident"] = base_incident

    rows: list[dict] = []
    for drv, g in h.groupby("driver_number", sort=False):
        tail = g.head(ir_n)
        n_used = len(tail)
        ir_pct = 100.0 * float(tail["incident"].mean()) if n_used > 0 else np.nan
        rows.append({"driver_number": int(drv), "ir_pct": ir_pct, "n_ir": n_used})
    return pd.DataFrame(rows)


def build_strategy_analysis(
    engine: Engine,
    session_id: int,
    cfg: StrategyAnalysisConfig = StrategyAnalysisConfig(),
) -> pd.DataFrame:
    race_df = _read_df(
        engine,
        """
        SELECT session_id, driver_number, team_name, finish_position, dnf, dns, dsq, number_of_laps
        FROM fact_race_result
        WHERE session_id = :sid
        """,
        {"sid": session_id},
        name="race_current",
        debug=cfg.debug,
        raise_on_error=cfg.raise_on_read_error,
    )
    if race_df.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "driver_number",
                "pit_count",
                "pit_median_ms",
                "strategy_type",
                "pit_loss_percentile",
                "ir_pct",
                "n_ir",
            ]
        )

    pit_df = _read_df(
        engine,
        """
        SELECT session_id, driver_number, stop_number, pit_ms
        FROM fact_pitstop
        WHERE session_id = :sid
        """,
        {"sid": session_id},
        name="pit_current",
        debug=cfg.debug,
        raise_on_error=cfg.raise_on_read_error,
    )

    history_df = pd.DataFrame()
    if cfg.enable_ir:
        history_df = _read_df(
            engine,
            """
            SELECT r.session_id, r.driver_number, r.dnf, r.dns, r.dsq, s.date_start
            FROM fact_race_result r
            JOIN dim_session s ON s.session_id = r.session_id
            WHERE s.date_start < (SELECT date_start FROM dim_session WHERE session_id = :sid)
              AND (s.session_type IS NULL OR LOWER(s.session_type) LIKE '%race%')
            ORDER BY s.date_start DESC
            """,
            {"sid": session_id},
            name="race_history_with_date_preview",
            debug=cfg.debug,
            raise_on_error=cfg.raise_on_read_error,
        )

    if cfg.debug:
        print(f"[DBG] sid={session_id} race_rows={len(race_df)} drivers={race_df['driver_number'].nunique()}")
        print(f"[DBG] pit_rows={len(pit_df)} pit_cols={list(pit_df.columns)}")
        if not pit_df.empty and "pit_ms" in pit_df.columns:
            null_rate = pd.to_numeric(pit_df["pit_ms"], errors="coerce").isna().mean()
            print(f"[DBG] pit_ms null_rate={null_rate:.3f}")
        elif not pit_df.empty and "pit_ms" not in pit_df.columns:
            print("[WARN] pit_df has no pit_ms column. Check ETL / schema.")
        hist_drv = history_df["driver_number"].nunique() if not history_df.empty and "driver_number" in history_df.columns else 0
        print(f"[DBG] hist_rows={len(history_df)} hist_drivers={hist_drv}")

    short_race = False
    if "number_of_laps" in race_df.columns and race_df["number_of_laps"].notna().any():
        laps = pd.to_numeric(race_df["number_of_laps"], errors="coerce")
        short_race = float(laps.median()) < cfg.short_race_lap_threshold

    pe = compute_pe_from_pit(pit_df, race_df)
    pe = classify_strategy_type(pe, short_race=short_race)
    pe = compute_pit_loss_percentile(pe, higher_is_worse=cfg.percentile_higher_is_worse)
    ir = pd.DataFrame(columns=["driver_number", "ir_pct", "n_ir"])
    if cfg.enable_ir:
        ir = compute_ir_pct_by_date(
            engine=engine,
            current_session_id=session_id,
            ir_n=cfg.ir_n,
            include_dns=cfg.include_dns_in_ir,
            debug=cfg.debug,
            raise_on_error=cfg.raise_on_read_error,
        )

    out = race_df[["driver_number"]].drop_duplicates().merge(pe, on="driver_number", how="left")
    out = out.merge(ir, on="driver_number", how="left")
    out["session_id"] = int(session_id)
    out = out[
        [
            "session_id",
            "driver_number",
            "pit_count",
            "pit_median_ms",
            "strategy_type",
            "pit_loss_percentile",
            "ir_pct",
            "n_ir",
        ]
    ].sort_values("driver_number")
    out["n_ir"] = out["n_ir"].fillna(0).astype(int)

    if cfg.save_table:
        with engine.begin() as conn:
            try:
                conn.execute(
                    text(f"DELETE FROM {cfg.save_table} WHERE session_id = :sid"),
                    {"sid": session_id},
                )
            except Exception:
                pass
        out.to_sql(cfg.save_table, engine, if_exists="append", index=False)

    return out.reset_index(drop=True)
