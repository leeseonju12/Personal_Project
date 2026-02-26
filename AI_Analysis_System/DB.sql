-- OpenF1 AI Analysis DB Schema (MySQL 8+)
-- Apply in your target DB (e.g. Spring_project_26_02)

SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS fact_race_result (
  session_id INT NOT NULL,
  driver_number INT NOT NULL,
  team_name VARCHAR(64) NULL,
  grid_position INT NULL,
  finish_position INT NULL,
  points DECIMAL(5,1) NULL,
  status VARCHAR(32) NULL,
  dnf TINYINT(1) NULL,
  dns TINYINT(1) NULL,
  dsq TINYINT(1) NULL,
  gap_to_leader VARCHAR(32) NULL,
  number_of_laps INT NULL,
  duration VARCHAR(32) NULL,
  PRIMARY KEY (session_id, driver_number),
  KEY ix_race_team_name (team_name),
  KEY ix_race_finish_position (finish_position)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS fact_lap (
  session_id INT NOT NULL,
  driver_number INT NOT NULL,
  lap_number INT NOT NULL,
  lap_time_ms INT NULL,
  is_pit_out_lap TINYINT(1) NULL,
  PRIMARY KEY (session_id, driver_number, lap_number),
  KEY ix_lap_session_driver (session_id, driver_number),
  KEY ix_lap_time (lap_time_ms)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS fact_pitstop (
  session_id INT NOT NULL,
  driver_number INT NOT NULL,
  stop_number INT NOT NULL,
  pit_ms INT NULL,
  PRIMARY KEY (session_id, driver_number, stop_number),
  KEY ix_pit_session_driver (session_id, driver_number),
  KEY ix_pit_ms (pit_ms)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS feat_driver_session_metrics (
  session_id INT NOT NULL,
  driver_number INT NOT NULL,
  qpi_pct DECIMAL(8,4) NULL,
  rpi_pct DECIMAL(8,4) NULL,
  clean_lap_median_ms INT NULL,
  clean_lap_count INT NULL,
  cs_sd_ms DECIMAL(10,3) NULL,
  cs_iqr_ms DECIMAL(10,3) NULL,
  pit_median_ms INT NULL,
  pit_count INT NULL,
  sfd INT NULL,
  metric_version VARCHAR(32) NOT NULL DEFAULT 'v1',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (session_id, driver_number),
  KEY ix_feat_driver (driver_number),
  KEY ix_feat_rpi (rpi_pct),
  KEY ix_feat_qpi (qpi_pct)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Optional storage tables for Monte Carlo forecast outputs
CREATE TABLE IF NOT EXISTS forecast_quali (
  id BIGINT NOT NULL AUTO_INCREMENT,
  target_session_id INT NOT NULL,
  driver_number INT NOT NULL,
  exp_qpi_mean DECIMAL(8,4) NULL,
  exp_qpi_sigma DECIMAL(8,4) NULL,
  pole_prob DECIMAL(8,5) NOT NULL,
  top3_prob DECIMAL(8,5) NOT NULL,
  top10_prob DECIMAL(8,5) NOT NULL,
  exp_grid_pos DECIMAL(8,3) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_forecast_quali (target_session_id, driver_number),
  KEY ix_forecast_quali_session (target_session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS forecast_race (
  id BIGINT NOT NULL AUTO_INCREMENT,
  target_session_id INT NOT NULL,
  driver_number INT NOT NULL,
  exp_points DECIMAL(8,3) NOT NULL,
  podium_prob DECIMAL(8,5) NOT NULL,
  top10_prob DECIMAL(8,5) NOT NULL,
  dnf_prob DECIMAL(8,5) NOT NULL,
  exp_finish_pos DECIMAL(8,3) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_forecast_race (target_session_id, driver_number),
  KEY ix_forecast_race_session (target_session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Backward-compatible patch for existing environments
ALTER TABLE feat_driver_session_metrics
  ADD COLUMN IF NOT EXISTS qpi_pct DECIMAL(8,4) NULL AFTER driver_number,
  ADD COLUMN IF NOT EXISTS metric_version VARCHAR(32) NOT NULL DEFAULT 'v1' AFTER sfd,
  ADD COLUMN IF NOT EXISTS created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  ADD COLUMN IF NOT EXISTS updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP;
