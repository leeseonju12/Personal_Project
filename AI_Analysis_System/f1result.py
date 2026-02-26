import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

BASE = "https://api.openf1.org/v1"
DB_URL = "mysql+pymysql://root@127.0.0.1:3306/Spring_project_26_02?charset=utf8mb4"

def get(endpoint: str, params: dict) -> list[dict]:
    r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def find_mexico_meeting_2025() -> dict:
    meetings = get("meetings", {"year": 2025, "country_name": "Mexico"})
    if not meetings:
        raise RuntimeError("OpenF1: 2025 Mexico meetings not found. (API에 데이터가 없을 수 있음)")
    # meeting_name에 Mexico 포함 우선
    meetings = sorted(meetings, key=lambda m: ("Mexico" not in (m.get("meeting_name") or ""), m.get("meeting_key")))
    return meetings[0]

def find_race_session(meeting_key: int) -> dict:
    sessions = get("sessions", {"meeting_key": meeting_key})
    if not sessions:
        raise RuntimeError(f"OpenF1: sessions not found for meeting_key={meeting_key}")
    for s in sessions:
        if (s.get("session_name") == "Race") or (s.get("session_type") == "Race"):
            return s
    for s in sessions:
        if "Race" in (s.get("session_name") or ""):
            return s
    raise RuntimeError("OpenF1: Race session not found")

def upsert_df(engine, sql: str, df: pd.DataFrame):
    if df.empty:
        return
    with engine.begin() as conn:
        conn.execute(text(sql), df.to_dict(orient="records"))

def compute_rpi_cs(laps: pd.DataFrame, min_clean_laps: int = 5) -> pd.DataFrame:
    if laps.empty:
        return pd.DataFrame()

    df = laps.copy()

    # OpenF1 laps: lap_duration(초) -> ms
    if "lap_duration" in df.columns:
        df["lap_time_ms"] = (df["lap_duration"].astype(float) * 1000).round().astype("Int64")
    elif "lap_time_ms" not in df.columns:
        raise RuntimeError("laps에 lap_duration이 없습니다. OpenF1 응답 컬럼을 확인하세요.")

    # 클린랩: pit out lap 제외(가능하면)
    if "is_pit_out_lap" in df.columns:
        df = df[df["is_pit_out_lap"] == False]  # noqa: E712

    df = df[df["lap_time_ms"].notna() & (df["lap_time_ms"] > 0)]

    rows = []
    for drv, part in df.groupby("driver_number"):
        x = part["lap_time_ms"].astype(int).to_numpy()
        n = len(x)
        if n < min_clean_laps:
            continue
        med = int(np.median(x))
        sd = float(np.std(x, ddof=1)) if n >= 2 else 0.0
        q75, q25 = np.percentile(x, [75, 25])
        iqr = float(q75 - q25)
        rows.append({
            "driver_number": int(drv),
            "clean_lap_median_ms": med,
            "clean_lap_count": n,
            "cs_sd_ms": sd,
            "cs_iqr_ms": iqr
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    fastest = out["clean_lap_median_ms"].min()
    out["rpi_pct"] = (out["clean_lap_median_ms"] / fastest - 1.0) * 100.0
    return out

def compute_pe(pit: pd.DataFrame) -> pd.DataFrame:
    if pit.empty:
        return pd.DataFrame()
    df = pit.copy()
    # OpenF1 pit: pit_duration(초) 가 일반적
    if "pit_duration" in df.columns:
        df["pit_ms"] = (df["pit_duration"].astype(float) * 1000).round().astype("Int64")
    elif "pit_ms" not in df.columns:
        raise RuntimeError("pit에 pit_duration이 없습니다. OpenF1 응답 컬럼을 확인하세요.")

    rows = []
    for drv, part in df.groupby("driver_number"):
        x = part["pit_ms"].dropna().astype(int).to_numpy()
        if len(x) == 0:
            continue
        rows.append({
            "driver_number": int(drv),
            "pit_median_ms": int(np.median(x)),
            "pit_count": int(len(x)),
        })
    return pd.DataFrame(rows)

def main():
    engine = create_engine(DB_URL, pool_pre_ping=True)

    meeting = find_mexico_meeting_2025()
    meeting_key = meeting["meeting_key"]
    race_session = find_race_session(meeting_key)
    session_key = race_session["session_key"]

    print(f"[OK] meeting={meeting.get('meeting_name')} meeting_key={meeting_key}")
    print(f"[OK] race session_key={session_key}")

    # --- fetch ---
    result = pd.DataFrame(get("session_result", {"session_key": session_key}))
    result = pd.DataFrame(get("session_result", {"session_key": session_key}))
    result["sfd"] = result["grid_position"] - result["position"]
    laps = pd.DataFrame(get("laps", {"session_key": session_key}))
    pit = pd.DataFrame(get("pit", {"session_key": session_key}))

    # --- normalize results ---
    # session_result: position, dnf/dns/dsq, gap_to_leader, number_of_laps, duration 등
    res = result.rename(columns={"position": "finish_position"}).copy()
    # grid: position = grid_position
    grd = grid.rename(columns={"position": "grid_position"}).copy()

    # 최소 컬럼
    res_small = res[[
        "driver_number", "finish_position", "dnf", "dns", "dsq",
        "gap_to_leader", "number_of_laps", "duration"
    ]].copy()

    if "points" in res.columns:
        res_small["points"] = res["points"]
    else:
        res_small["points"] = None

    # status를 만들고 싶으면: dnf/dns/dsq 기반 MVP
    def status_row(r):
        if bool(r.get("dsq")): return "DSQ"
        if bool(r.get("dns")): return "DNS"
        if bool(r.get("dnf")): return "DNF"
        return "Finished"
    res_small["status"] = res_small.apply(status_row, axis=1)

    grd_small = grd[["driver_number", "grid_position"]].copy()

    merged = res_small.merge(grd_small, on="driver_number", how="left")
    merged["session_id"] = int(session_key)
    merged["sfd"] = merged["grid_position"] - merged["finish_position"]

    # team_name이 결과에 있으면 저장(없으면 NULL)
    merged["team_name"] = res.get("team_name") if "team_name" in res.columns else None

    # --- save fact_race_result ---
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
    upsert_df(engine, upsert_race_sql, merged)

    # --- save fact_lap (최소) ---
    if not laps.empty:
        lap_df = laps.copy()
        lap_df["session_id"] = int(session_key)
        lap_df["lap_time_ms"] = (lap_df["lap_duration"].astype(float) * 1000).round().astype("Int64")
        if "is_pit_out_lap" not in lap_df.columns:
            lap_df["is_pit_out_lap"] = None

        lap_small = lap_df[["session_id", "driver_number", "lap_number", "lap_time_ms", "is_pit_out_lap"]].copy()

        upsert_lap_sql = """
        INSERT INTO fact_lap
        (session_id, driver_number, lap_number, lap_time_ms, is_pit_out_lap)
        VALUES (:session_id, :driver_number, :lap_number, :lap_time_ms, :is_pit_out_lap)
        ON DUPLICATE KEY UPDATE
          lap_time_ms=VALUES(lap_time_ms),
          is_pit_out_lap=VALUES(is_pit_out_lap);
        """
        upsert_df(engine, upsert_lap_sql, lap_small)

    # --- save fact_pitstop (최소) ---
    if not pit.empty:
        pit_df = pit.copy()
        pit_df["session_id"] = int(session_key)
        pit_df["pit_ms"] = (pit_df["pit_duration"].astype(float) * 1000).round().astype("Int64")
        # OpenF1 pit stop number 컬럼명이 다를 수 있어 stop_number가 없으면 index로 생성
        if "stop_number" not in pit_df.columns:
            pit_df["stop_number"] = pit_df.groupby("driver_number").cumcount() + 1

        pit_small = pit_df[["session_id","driver_number","stop_number","pit_ms"]].copy()

        upsert_pit_sql = """
        INSERT INTO fact_pitstop
        (session_id, driver_number, stop_number, pit_ms)
        VALUES (:session_id, :driver_number, :stop_number, :pit_ms)
        ON DUPLICATE KEY UPDATE
          pit_ms=VALUES(pit_ms);
        """
        upsert_df(engine, upsert_pit_sql, pit_small)

    # --- compute metrics ---
    rpi_cs = compute_rpi_cs(laps)
    pe = compute_pe(pit)

    metrics = merged[["session_id","driver_number","sfd"]].copy()
    metrics = metrics.merge(rpi_cs, on="driver_number", how="left").merge(pe, on="driver_number", how="left")

    # --- save feat_driver_session_metrics ---
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
    upsert_df(engine, upsert_feat_sql, metrics)

    print("\n✅ DONE: Saved fact_race_result, fact_lap, fact_pitstop, feat_driver_session_metrics")
    print("Top 10 (finish + rpi):")
    show = metrics.merge(merged[["driver_number","finish_position","grid_position"]], on="driver_number", how="left")
    show = show.sort_values("finish_position")[["finish_position","driver_number","grid_position","sfd","rpi_pct","cs_sd_ms","pit_median_ms"]]
    print(show.head(10).to_string(index=False))

if __name__ == "__main__":
    main()