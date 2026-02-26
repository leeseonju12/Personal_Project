# =============================
# OpenF1 - python DB

CREATE TABLE IF NOT EXISTS fact_race_result (
  session_id INT NOT NULL,
  driver_number INT NOT NULL,
  team_name VARCHAR(64) NULL,
  grid_position INT NULL,
  finish_position INT NULL,
  points DECIMAL(4,1) NULL,
  STATUS VARCHAR(32) NULL,
  dnf TINYINT NULL,
  dns TINYINT NULL,
  dsq TINYINT NULL,
  gap_to_leader VARCHAR(32) NULL,
  number_of_laps INT NULL,
  duration VARCHAR(32) NULL,
  PRIMARY KEY (session_id, driver_number)
);

CREATE TABLE IF NOT EXISTS fact_lap (
  session_id INT NOT NULL,
  driver_number INT NOT NULL,
  lap_number INT NOT NULL,
  lap_time_ms INT NULL,
  is_pit_out_lap TINYINT NULL,
  PRIMARY KEY (session_id, driver_number, lap_number)
);

CREATE TABLE IF NOT EXISTS fact_pitstop (
  session_id INT NOT NULL,
  driver_number INT NOT NULL,
  stop_number INT NOT NULL,
  pit_ms INT NULL,
  PRIMARY KEY (session_id, driver_number, stop_number)
);

-- 결과 지표 저장(최소)
CREATE TABLE IF NOT EXISTS feat_driver_session_metrics (
  session_id INT NOT NULL,
  driver_number INT NOT NULL,
  rpi_pct DECIMAL(7,4) NULL,
  clean_lap_median_ms INT NULL,
  clean_lap_count INT NULL,
  cs_sd_ms DECIMAL(8,3) NULL,
  cs_iqr_ms DECIMAL(8,3) NULL,
  pit_median_ms INT NULL,
  pit_count INT NULL,
  sfd INT NULL,
  PRIMARY KEY (session_id, driver_number)
);

