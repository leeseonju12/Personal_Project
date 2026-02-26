import os
from typing import Any

import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine, text

BASE_URL = "https://api.openf1.org/v1"
DB_URL = os.getenv(
    "F1_DB_URL",
    "mysql+pymysql://root@127.0.0.1:3306/Spring_project_26_02?charset=utf8mb4",
)
TARGET_YEAR = int(os.getenv("F1_YEAR", "2025"))
TARGET_COUNTRY = os.getenv("F1_COUNTRY", "Mexico")
MIN_CLEAN_LAPS = int(os.getenv("F1_MIN_CLEAN_LAPS", "5"))

def upsert_df(engine, sql: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    # MySQL does not accept NaN; convert missing values to NULL.
    clean_df = df.astype(object).where(pd.notnull(df), None)
    with engine.begin() as conn:
        conn.execute(text(sql), clean_df.to_dict(orient="records"))

def get_json(endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected OpenF1 response type for {endpoint}: {type(data)}")
    return data


def find_meeting(year: int, country_name: str) -> dict[str, Any]:
    meetings = get_json("meetings", {"year": year, "country_name": country_name})
    if not meetings:
        raise RuntimeError(f"Meeting not found: year={year}, country={country_name}")
    meetings = sorted(
        meetings,
        key=lambda m: (
            country_name.lower() not in (m.get("meeting_name") or "").lower(),
            m.get("meeting_key", 999999),
        ),
    )
    return meetings[0]


def find_session(meeting_key: int, keywords: tuple[str, ...]) -> dict[str, Any] | None:
    sessions = get_json("sessions", {"meeting_key": meeting_key})
    for s in sessions:
        name = (s.get("session_name") or "").lower()
        s_type = (s.get("session_type") or "").lower()
        if any(k in name or k in s_type for k in keywords):
            return s
    return None


def _normalize_lap_time_ms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "lap_time_ms" in out.columns:
        return out
    if "lap_duration" not in out.columns:
        raise RuntimeError("laps payload has no lap_duration")
    out["lap_time_ms"] = (out["lap_duration"].astype(float) * 1000.0).round().astype("Int64")
    return out


def _normalize_pit_ms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pit_ms" in out.columns:
        return out
    if "pit_duration" not in out.columns:
        raise RuntimeError("pit payload has no pit_duration")
    out["pit_ms"] = (out["pit_duration"].astype(float) * 1000.0).round().astype("Int64")
    return out


def _to_bool_int(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool).astype(int)


def clean_race_laps(laps: pd.DataFrame) -> pd.DataFrame:
    if laps.empty:
        return laps

    df = _normalize_lap_time_ms(laps)
    df = df[df["lap_time_ms"].notna() & (df["lap_time_ms"] > 0)]

    if "is_pit_out_lap" in df.columns:
        df = df[df["is_pit_out_lap"] != True]  # noqa: E712
    if "is_pit_in_lap" in df.columns:
        df = df[df["is_pit_in_lap"] != True]  # noqa: E712
    if "track_status" in df.columns:
        df = df[~df["track_status"].fillna("").isin(["SC", "VSC"]) ]

    return df


def compute_qpi(quali_laps: pd.DataFrame) -> pd.DataFrame:
    if quali_laps.empty:
        return pd.DataFrame(columns=["driver_number", "best_lap_ms", "qpi_pct"])

    df = _normalize_lap_time_ms(quali_laps)
    best = (
        df[df["lap_time_ms"].notna() & (df["lap_time_ms"] > 0)]
        .groupby("driver_number", as_index=False)["lap_time_ms"]
        .min()
        .rename(columns={"lap_time_ms": "best_lap_ms"})
    )

    if best.empty:
        return pd.DataFrame(columns=["driver_number", "best_lap_ms", "qpi_pct"])

    session_best = best["best_lap_ms"].min()
    best["qpi_pct"] = (best["best_lap_ms"] / session_best - 1.0) * 100.0
    return best


def compute_rpi_cs(race_laps: pd.DataFrame, min_clean_laps: int = 5) -> pd.DataFrame:
    if race_laps.empty:
        return pd.DataFrame(
            columns=["driver_number", "clean_lap_median_ms", "clean_lap_count", "cs_sd_ms", "cs_iqr_ms", "rpi_pct"]
        )

    df = clean_race_laps(race_laps)
    rows: list[dict[str, Any]] = []

    for driver_number, part in df.groupby("driver_number"):
        x = part["lap_time_ms"].dropna().astype(int).to_numpy()
        if len(x) < min_clean_laps:
            continue
        rows.append(
            {
                "driver_number": int(driver_number),
                "clean_lap_median_ms": int(np.median(x)),
                "clean_lap_count": int(len(x)),
                "cs_sd_ms": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                "cs_iqr_ms": float(np.percentile(x, 75) - np.percentile(x, 25)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=["driver_number", "clean_lap_median_ms", "clean_lap_count", "cs_sd_ms", "cs_iqr_ms", "rpi_pct"]
        )

    fastest = out["clean_lap_median_ms"].min()
    out["rpi_pct"] = (out["clean_lap_median_ms"] / fastest - 1.0) * 100.0
    return out


def compute_pe(pit: pd.DataFrame) -> pd.DataFrame:
    if pit.empty:
        return pd.DataFrame(columns=["driver_number", "pit_median_ms", "pit_count"])

    df = _normalize_pit_ms(pit)
    rows: list[dict[str, Any]] = []

    for driver_number, part in df.groupby("driver_number"):
        x = part["pit_ms"].dropna().astype(int).to_numpy()
        if len(x) == 0:
            continue
        rows.append(
            {
                "driver_number": int(driver_number),
                "pit_median_ms": int(np.median(x)),
                "pit_count": int(len(x)),
            }
        )

    return pd.DataFrame(rows)


def build_fact_race_result(session_id: int, result_raw: pd.DataFrame) -> pd.DataFrame:
    if result_raw.empty:
        return pd.DataFrame()

    res = result_raw.copy()
    if "position" in res.columns and "finish_position" not in res.columns:
        res = res.rename(columns={"position": "finish_position"})

    for col in ["dnf", "dns", "dsq"]:
        if col not in res.columns:
            res[col] = False

    for col in ["points", "team_name", "gap_to_leader", "number_of_laps", "duration", "grid_position", "finish_position"]:
        if col not in res.columns:
            res[col] = None

    status = np.where(
        _to_bool_int(res["dsq"]) == 1,
        "DSQ",
        np.where(_to_bool_int(res["dns"]) == 1, "DNS", np.where(_to_bool_int(res["dnf"]) == 1, "DNF", "Finished")),
    )

    out = pd.DataFrame(
        {
            "session_id": int(session_id),
            "driver_number": res["driver_number"],
            "team_name": res["team_name"],
            "grid_position": res["grid_position"],
            "finish_position": res["finish_position"],
            "points": res["points"],
            "status": status,
            "dnf": _to_bool_int(res["dnf"]),
            "dns": _to_bool_int(res["dns"]),
            "dsq": _to_bool_int(res["dsq"]),
            "gap_to_leader": res["gap_to_leader"],
            "number_of_laps": res["number_of_laps"],
            "duration": res["duration"],
        }
    )

    out["sfd"] = pd.to_numeric(out["grid_position"], errors="coerce") - pd.to_numeric(
        out["finish_position"], errors="coerce"
    )
    return out


def build_fact_lap(session_id: int, laps_raw: pd.DataFrame) -> pd.DataFrame:
    if laps_raw.empty:
        return pd.DataFrame()

    df = _normalize_lap_time_ms(laps_raw)
    if "is_pit_out_lap" not in df.columns:
        df["is_pit_out_lap"] = None

    out = df[["driver_number", "lap_number", "lap_time_ms", "is_pit_out_lap"]].copy()
    out.insert(0, "session_id", int(session_id))
    return out


def build_fact_pitstop(session_id: int, pit_raw: pd.DataFrame) -> pd.DataFrame:
    if pit_raw.empty:
        return pd.DataFrame()

    df = _normalize_pit_ms(pit_raw)
    if "stop_number" not in df.columns:
        df["stop_number"] = df.groupby("driver_number").cumcount() + 1

    out = df[["driver_number", "stop_number", "pit_ms"]].copy()
    out.insert(0, "session_id", int(session_id))
    return out


def build_feat_metrics(session_id: int, fact_race: pd.DataFrame, race_laps: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    base = fact_race[["driver_number", "sfd"]].copy()
    base.insert(0, "session_id", int(session_id))

    rpi_cs = compute_rpi_cs(race_laps, min_clean_laps=MIN_CLEAN_LAPS)
    pe = compute_pe(pit)

    metrics = base.merge(rpi_cs, on="driver_number", how="left").merge(pe, on="driver_number", how="left")
    return metrics


def simulate_quali_probabilities(qpi_df: pd.DataFrame, n_sims: int = 2000, seed: int = 42) -> pd.DataFrame:
    if qpi_df.empty:
        return pd.DataFrame(columns=["driver_number", "pole_prob", "top3_prob", "top10_prob", "exp_grid_pos"])

    rng = np.random.default_rng(seed)
    drivers = qpi_df["driver_number"].to_numpy()
    mu = qpi_df["qpi_pct"].to_numpy(dtype=float)
    sigma = np.maximum(qpi_df["qpi_pct"].std(ddof=0) * 0.6, 0.05)

    samples = rng.normal(mu, sigma, size=(n_sims, len(drivers)))
    ranks = np.argsort(np.argsort(samples, axis=1), axis=1) + 1

    rows = []
    for idx, drv in enumerate(drivers):
        r = ranks[:, idx]
        rows.append(
            {
                "driver_number": int(drv),
                "pole_prob": float((r == 1).mean()),
                "top3_prob": float((r <= 3).mean()),
                "top10_prob": float((r <= 10).mean()),
                "exp_grid_pos": float(r.mean()),
            }
        )
    return pd.DataFrame(rows)


def simulate_race_points(metrics_df: pd.DataFrame, n_sims: int = 2000, seed: int = 42) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["driver_number", "exp_points", "podium_prob", "top10_prob", "dnf_prob"])

    rng = np.random.default_rng(seed)
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

    df = metrics_df.copy()
    for col in ["rpi_pct", "cs_iqr_ms", "pit_median_ms", "sfd"]:
        if col not in df.columns:
            df[col] = 0.0

    def zscore(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    strength = (
        0.45 * (-zscore(df["rpi_pct"].fillna(df["rpi_pct"].median())))
        + 0.20 * (-zscore(df["cs_iqr_ms"].fillna(df["cs_iqr_ms"].median())))
        + 0.15 * (-zscore(df["pit_median_ms"].fillna(df["pit_median_ms"].median())))
        + 0.20 * (zscore(df["sfd"].fillna(0.0)))
    )

    driver_numbers = df["driver_number"].to_numpy()
    n_driver = len(driver_numbers)

    finish_positions = np.zeros((n_sims, n_driver), dtype=int)
    points_scored = np.zeros((n_sims, n_driver), dtype=float)

    dnf_prob = np.clip(0.03 + 0.01 * np.maximum(0, -strength.to_numpy()), 0.02, 0.15)

    for sim in range(n_sims):
        perf = rng.normal(strength.to_numpy(), 0.35)
        dnf_flags = rng.random(n_driver) < dnf_prob

        alive_idx = np.where(~dnf_flags)[0]
        dead_idx = np.where(dnf_flags)[0]

        order_alive = alive_idx[np.argsort(-perf[alive_idx])]

        fp = np.empty(n_driver, dtype=int)
        for pos, i in enumerate(order_alive, start=1):
            fp[i] = pos

        if len(dead_idx) > 0:
            rng.shuffle(dead_idx)
            for pos, i in enumerate(dead_idx, start=len(order_alive) + 1):
                fp[i] = pos

        finish_positions[sim] = fp
        for i in range(n_driver):
            points_scored[sim, i] = points_map.get(int(fp[i]), 0)

    rows = []
    for i, drv in enumerate(driver_numbers):
        fp = finish_positions[:, i]
        rows.append(
            {
                "driver_number": int(drv),
                "exp_points": float(points_scored[:, i].mean()),
                "podium_prob": float((fp <= 3).mean()),
                "top10_prob": float((fp <= 10).mean()),
                "dnf_prob": float(dnf_prob[i]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    engine = create_engine(DB_URL, pool_pre_ping=True)

    meeting = find_meeting(TARGET_YEAR, TARGET_COUNTRY)
    meeting_key = int(meeting["meeting_key"])

    race_session = find_session(meeting_key, ("race",))
    if race_session is None:
        raise RuntimeError("Race session not found")
    race_session_key = int(race_session["session_key"])

    quali_session = find_session(meeting_key, ("qualifying", "quali"))
    quali_session_key = int(quali_session["session_key"]) if quali_session else None

    print(f"[INFO] meeting={meeting.get('meeting_name')} meeting_key={meeting_key}")
    print(f"[INFO] race_session_key={race_session_key}, quali_session_key={quali_session_key}")

    race_result_raw = pd.DataFrame(get_json("session_result", {"session_key": race_session_key}))
    race_laps_raw = pd.DataFrame(get_json("laps", {"session_key": race_session_key}))
    race_pit_raw = pd.DataFrame(get_json("pit", {"session_key": race_session_key}))

    quali_laps_raw = (
        pd.DataFrame(get_json("laps", {"session_key": quali_session_key})) if quali_session_key is not None else pd.DataFrame()
    )

    fact_race = build_fact_race_result(race_session_key, race_result_raw)
    fact_lap = build_fact_lap(race_session_key, race_laps_raw)
    fact_pit = build_fact_pitstop(race_session_key, race_pit_raw)
    feat_metrics = build_feat_metrics(race_session_key, fact_race, race_laps_raw, race_pit_raw)

    upsert_race_sql = """
    INSERT INTO fact_race_result
    (session_id, driver_number, team_name, grid_position, finish_position, points, status,
     dnf, dns, dsq, gap_to_leader, number_of_laps, duration)
    VALUES
    (:session_id, :driver_number, :team_name, :grid_position, :finish_position, :points, :status,
     :dnf, :dns, :dsq, :gap_to_leader, :number_of_laps, :duration)
    ON DUPLICATE KEY UPDATE
      team_name=VALUES(team_name),
      grid_position=VALUES(grid_position),
      finish_position=VALUES(finish_position),
      points=VALUES(points),
      status=VALUES(status),
      dnf=VALUES(dnf), dns=VALUES(dns), dsq=VALUES(dsq),
      gap_to_leader=VALUES(gap_to_leader),
      number_of_laps=VALUES(number_of_laps),
      duration=VALUES(duration);
    """

    upsert_lap_sql = """
    INSERT INTO fact_lap
    (session_id, driver_number, lap_number, lap_time_ms, is_pit_out_lap)
    VALUES (:session_id, :driver_number, :lap_number, :lap_time_ms, :is_pit_out_lap)
    ON DUPLICATE KEY UPDATE
      lap_time_ms=VALUES(lap_time_ms),
      is_pit_out_lap=VALUES(is_pit_out_lap);
    """

    upsert_pit_sql = """
    INSERT INTO fact_pitstop
    (session_id, driver_number, stop_number, pit_ms)
    VALUES (:session_id, :driver_number, :stop_number, :pit_ms)
    ON DUPLICATE KEY UPDATE
      pit_ms=VALUES(pit_ms);
    """

    upsert_feat_sql = """
    INSERT INTO feat_driver_session_metrics
    (session_id, driver_number, rpi_pct, clean_lap_median_ms, clean_lap_count,
     cs_sd_ms, cs_iqr_ms, pit_median_ms, pit_count, sfd)
    VALUES
    (:session_id, :driver_number, :rpi_pct, :clean_lap_median_ms, :clean_lap_count,
     :cs_sd_ms, :cs_iqr_ms, :pit_median_ms, :pit_count, :sfd)
    ON DUPLICATE KEY UPDATE
      rpi_pct=VALUES(rpi_pct),
      clean_lap_median_ms=VALUES(clean_lap_median_ms),
      clean_lap_count=VALUES(clean_lap_count),
      cs_sd_ms=VALUES(cs_sd_ms),
      cs_iqr_ms=VALUES(cs_iqr_ms),
      pit_median_ms=VALUES(pit_median_ms),
      pit_count=VALUES(pit_count),
      sfd=VALUES(sfd);
    """

    upsert_df(engine, upsert_race_sql, fact_race)
    upsert_df(engine, upsert_lap_sql, fact_lap)
    upsert_df(engine, upsert_pit_sql, fact_pit)
    upsert_df(engine, upsert_feat_sql, feat_metrics)

    qpi_df = compute_qpi(quali_laps_raw)
    quali_mc = simulate_quali_probabilities(qpi_df)
    race_mc = simulate_race_points(feat_metrics)

    summary = fact_race[["driver_number", "grid_position", "finish_position", "sfd"]].merge(
        feat_metrics[["driver_number", "rpi_pct", "cs_iqr_ms", "pit_median_ms"]],
        on="driver_number",
        how="left",
    )
    summary = summary.merge(qpi_df[["driver_number", "qpi_pct"]], on="driver_number", how="left")
    summary = summary.sort_values("finish_position")

    print("\n[DONE] Saved: fact_race_result, fact_lap, fact_pitstop, feat_driver_session_metrics")
    print("\nTop 10 Summary")
    print(summary.head(10).to_string(index=False))

    if not quali_mc.empty:
        print("\nQuali Forecast (MC)")
        print(quali_mc.sort_values("exp_grid_pos").head(10).to_string(index=False))

    if not race_mc.empty:
        print("\nRace Forecast (MC)")
        print(race_mc.sort_values("exp_points", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
