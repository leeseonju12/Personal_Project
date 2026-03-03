-- ============================================================
--  F1 Analytics  –  Spring_project_26_02  DB Schema
--  Last updated : 2026-03-03
--  Changes:
--    1. pit_ms / pit_median_ms / clean_lap_median_ms → DECIMAL(10,1)
--    2. fact_race_result nullable columns relaxed
--    3. Removed redundant ALTER TABLE block (already in CREATE TABLE)
--    4. fact_lap retained with comment (no save logic yet)
--    5. Added created_at indexes on forecast_quali / forecast_race
--    6. Added v_latest_dashboard VIEW for Framer API
-- ============================================================

DROP DATABASE IF EXISTS `Spring_project_26_02`;
CREATE DATABASE `Spring_project_26_02`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE `Spring_project_26_02`;

SET NAMES utf8mb4;

-- ------------------------------------------------------------
-- 1. dim_session  (기준 테이블이므로 먼저 생성)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_session (
  session_id   INT          NOT NULL,
  meeting_key  INT          NULL,
  YEAR         INT          NULL,
  round_no     INT          NULL,
  session_name VARCHAR(64)  NULL,
  session_type VARCHAR(32)  NULL,
  country_name VARCHAR(64)  NULL,
  meeting_name VARCHAR(128) NULL,
  date_start   DATETIME     NULL,
  date_end     DATETIME     NULL,
  PRIMARY KEY (session_id),
  KEY ix_dim_session_date_start (date_start),
  KEY ix_dim_session_year       (YEAR)
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- 2. fact_race_result
--    · grid_position / finish_position / number_of_laps → NULL 허용
--      (Python Int64 Nullable 타입 대응)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_race_result (
  session_id      INT           NOT NULL,
  driver_number   INT           NOT NULL,
  team_name       VARCHAR(64)   NULL,
  grid_position   INT           NULL,          -- ✅ NULL 허용으로 변경
  finish_position INT           NULL,          -- ✅ NULL 허용으로 변경
  points          DECIMAL(5,1)  NULL,
  STATUS          VARCHAR(32)   NULL,
  dnf             TINYINT(1)    NULL,
  dns             TINYINT(1)    NULL,
  dsq             TINYINT(1)    NULL,
  gap_to_leader   VARCHAR(32)   NULL,
  number_of_laps  INT           NULL,          -- ✅ NULL 허용으로 변경
  duration        VARCHAR(32)   NULL,
  PRIMARY KEY (session_id, driver_number),
  KEY ix_race_team_name      (team_name),
  KEY ix_race_finish_position (finish_position)
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- 3. fact_lap
--    · 현재 Python 저장 로직 없음 → 스키마만 유지
--      (향후 save_outputs_to_db 에서 laps 저장 시 활성화)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_lap (
  session_id     INT        NOT NULL,
  driver_number  INT        NOT NULL,
  lap_number     INT        NOT NULL,
  lap_time_ms    INT        NULL,
  is_pit_out_lap TINYINT(1) NULL,
  PRIMARY KEY (session_id, driver_number, lap_number),
  KEY ix_lap_session_driver (session_id, driver_number),
  KEY ix_lap_time           (lap_time_ms)
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- 4. fact_pitstop
--    · pit_ms : float 소수점 손실 방지 → DECIMAL(10,1) ✅
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_pitstop (
  session_id    INT           NOT NULL,
  driver_number INT           NOT NULL,
  stop_number   INT           NOT NULL,
  pit_ms        DECIMAL(10,1) NULL,            -- ✅ INT → DECIMAL(10,1)
  PRIMARY KEY (session_id, driver_number, stop_number),
  KEY ix_pit_session_driver (session_id, driver_number),
  KEY ix_pit_ms             (pit_ms)
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- 5. feat_driver_session_metrics
--    · pit_median_ms / clean_lap_median_ms → DECIMAL(10,1) ✅
--    · metric_version / created_at / updated_at 처음부터 포함
--      (ALTER TABLE 블록 제거)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feat_driver_session_metrics (
  session_id         INT           NOT NULL,
  driver_number      INT           NOT NULL,
  qpi_pct            DECIMAL(8,4)  NULL,
  rpi_pct            DECIMAL(8,4)  NULL,
  clean_lap_median_ms DECIMAL(10,1) NULL,       -- ✅ INT → DECIMAL(10,1)
  clean_lap_count    INT           NULL,
  cs_sd_ms           DECIMAL(10,3) NULL,
  cs_iqr_ms          DECIMAL(10,3) NULL,
  pit_median_ms      DECIMAL(10,1) NULL,        -- ✅ INT → DECIMAL(10,1)
  pit_count          INT           NULL,
  sfd                INT           NULL,
  metric_version     VARCHAR(32)   NOT NULL DEFAULT 'v1',
  created_at         DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at         DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
                                   ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (session_id, driver_number),
  KEY ix_feat_driver (driver_number),
  KEY ix_feat_rpi    (rpi_pct),
  KEY ix_feat_qpi    (qpi_pct)
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- 6. forecast_quali
--    · created_at 인덱스 추가 ✅
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecast_quali (
  id                BIGINT        NOT NULL AUTO_INCREMENT,
  target_session_id INT           NOT NULL,
  driver_number     INT           NOT NULL,
  exp_qpi_mean      DECIMAL(8,4)  NULL,
  exp_qpi_sigma     DECIMAL(8,4)  NULL,
  pole_prob         DECIMAL(8,5)  NOT NULL,
  top3_prob         DECIMAL(8,5)  NOT NULL,
  top10_prob        DECIMAL(8,5)  NOT NULL,
  exp_grid_pos      DECIMAL(8,3)  NOT NULL,
  created_at        DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_forecast_quali      (target_session_id, driver_number),
  KEY ix_forecast_quali_session     (target_session_id),
  KEY ix_forecast_quali_created     (created_at)               -- ✅ 추가
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- 7. forecast_race
--    · created_at 인덱스 추가 ✅
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecast_race (
  id                BIGINT        NOT NULL AUTO_INCREMENT,
  target_session_id INT           NOT NULL,
  driver_number     INT           NOT NULL,
  exp_points        DECIMAL(8,3)  NOT NULL,
  podium_prob       DECIMAL(8,5)  NOT NULL,
  top10_prob        DECIMAL(8,5)  NOT NULL,
  dnf_prob          DECIMAL(8,5)  NOT NULL,
  exp_finish_pos    DECIMAL(8,3)  NULL,
  created_at        DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_forecast_race       (target_session_id, driver_number),
  KEY ix_forecast_race_session      (target_session_id),
  KEY ix_forecast_race_created      (created_at)               -- ✅ 추가
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- 8. analysis_strategy
--    · pit_median_ms → DECIMAL(10,1) ✅
--    · delete_specs 에 포함되므로 재실행 시 중복 방지 가능
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analysis_strategy (
  session_id           INT           NOT NULL,
  driver_number        INT           NOT NULL,
  pit_count            INT           NULL,
  pit_median_ms        DECIMAL(10,1) NULL,      -- ✅ INT → DECIMAL(10,1)
  strategy_type        VARCHAR(32)   NULL,
  pit_loss_percentile  DECIMAL(8,3)  NULL,
  ir_pct               DECIMAL(8,3)  NULL,
  n_ir                 INT           NULL,
  PRIMARY KEY (session_id, driver_number),
  KEY ix_analysis_strategy_session (session_id),
  KEY ix_analysis_strategy_driver  (driver_number)
) ENGINE=INNODB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- VIEW : v_latest_dashboard
--   Framer API에서 JOIN 없이 최신 세션 전체 데이터를 한 번에 조회
--   FastAPI endpoint : GET /api/dashboard
-- ============================================================
CREATE OR REPLACE VIEW v_latest_dashboard AS
SELECT
    fq.target_session_id                          AS session_id,
    fq.driver_number,
    fq.exp_grid_pos,
    fq.pole_prob,
    fq.top3_prob,
    fq.top10_prob                                 AS quali_top10_prob,
    fr.exp_points,
    fr.podium_prob,
    fr.top10_prob                                 AS race_top10_prob,
    fr.dnf_prob,
    fr.exp_finish_pos,
    ast.strategy_type,
    ast.pit_count,
    ast.pit_median_ms,
    ast.pit_loss_percentile,
    ast.ir_pct,
    ds.country_name,
    ds.meeting_name,
    ds.date_start,
    ds.round_no,
    ds.year
FROM forecast_quali fq
JOIN forecast_race fr
    ON  fq.target_session_id = fr.target_session_id
    AND fq.driver_number     = fr.driver_number
LEFT JOIN analysis_strategy ast
    ON  fq.target_session_id = ast.session_id
    AND fq.driver_number     = ast.driver_number
LEFT JOIN dim_session ds
    ON  fq.target_session_id = ds.session_id
WHERE fq.target_session_id = (
    SELECT MAX(target_session_id) FROM forecast_race
)
ORDER BY fr.exp_points DESC;

-- ============================================================
-- 결과 조회 확인용 쿼리
-- ============================================================
SELECT * FROM forecast_quali    ORDER BY target_session_id DESC, exp_grid_pos         ASC;
SELECT * FROM analysis_strategy ORDER BY session_id        DESC, pit_loss_percentile  DESC;
SELECT * FROM forecast_race     ORDER BY target_session_id DESC, exp_points           DESC;
SELECT * FROM v_latest_dashboard;