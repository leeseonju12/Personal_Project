USE `Spring_project_26_02`;

-- ============================================================
-- 0) 안전한 세션 변수 초기화
-- ============================================================
SET @LATEST_SESSION_ID := NULL;
SET @LATEST_TARGET_SESSION_ID := NULL;

-- feat_driver_session_metrics 쪽 최신 session
SELECT COALESCE(
               (SELECT MAX(session_id) FROM feat_driver_session_metrics),
               (SELECT MAX(session_id) FROM dim_session)
       )
INTO @LATEST_SESSION_ID;

-- forecast_quali 쪽 최신 target session
SELECT COALESCE(
               (SELECT MAX(target_session_id) FROM forecast_quali),
               (SELECT MAX(session_id) FROM dim_session)
       )
INTO @LATEST_TARGET_SESSION_ID;

-- ============================================================
-- 1) 드라이버 차원 테이블
-- ============================================================
CREATE TABLE IF NOT EXISTS dim_driver (
                                          driver_number INT NOT NULL,
                                          driver_name   VARCHAR(64) NOT NULL,
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (driver_number),
    UNIQUE KEY uq_dim_driver_name (driver_name)
    ) ENGINE=INNODB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- 2) 컬럼 추가 (이미 있으면 건너뜀)
--    Duplicate column 오류 방지
-- ============================================================
ALTER TABLE feat_driver_session_metrics
    ADD COLUMN IF NOT EXISTS best_lap_ms DECIMAL(10,1) NULL AFTER driver_number;

ALTER TABLE forecast_quali
    ADD COLUMN IF NOT EXISTS starting_grid_position INT NULL AFTER driver_number;

ALTER TABLE dim_driver
    ADD COLUMN IF NOT EXISTS team_name VARCHAR(64) NULL AFTER driver_name;

-- ============================================================
-- 3) 드라이버명 UPSERT
-- ============================================================
INSERT INTO dim_driver (driver_number, driver_name, team_name) VALUES
                                                                   (4,  'Lando NORRIS',        'McLaren'),
                                                                   (16, 'Charles LECLERC',     'Ferrari'),
                                                                   (44, 'Lewis HAMILTON',      'Ferrari'),
                                                                   (63, 'George RUSSELL',      'Mercedes'),
                                                                   (1,  'Max VERSTAPPEN',      'Red Bull Racing'),
                                                                   (12, 'Kimi ANTONELLI',      'Mercedes'),
                                                                   (55, 'Carlos SAINZ',        'Williams'),
                                                                   (81, 'Oscar PIASTRI',       'McLaren'),
                                                                   (6,  'Isack HADJAR',        'Racing Bulls'),
                                                                   (87, 'Oliver BEARMAN',      'Haas F1 Team'),
                                                                   (31, 'Esteban OCON',        'Haas F1 Team'),
                                                                   (18, 'Lance STROLL',        'Aston Martin'),
                                                                   (30, 'Liam LAWSON',         'Racing Bulls'),
                                                                   (22, 'Yuki TSUNODA',        'Red Bull Racing'),
                                                                   (14, 'Fernando ALONSO',     'Aston Martin'),
                                                                   (10, 'Pierre GASLY',        'Alpine'),
                                                                   (43, 'Franco COLAPINTO',    'Alpine')
    ON DUPLICATE KEY UPDATE
                         driver_name = VALUES(driver_name),
                         team_name   = VALUES(team_name),
                         updated_at  = CURRENT_TIMESTAMP;

-- ============================================================
-- 4) best_lap_ms / qpi_pct 반영
--    @LATEST_SESSION_ID 가 NULL이면 아무 행도 안 들어감
-- ============================================================
INSERT INTO feat_driver_session_metrics (
    session_id, driver_number, best_lap_ms, qpi_pct, metric_version
)
SELECT t.session_id, t.driver_number, t.best_lap_ms, t.qpi_pct, 'v1'
FROM (
         SELECT @LATEST_SESSION_ID AS session_id, 4  AS driver_number, 75586.0 AS best_lap_ms, 0.0000 AS qpi_pct
         UNION ALL SELECT @LATEST_SESSION_ID, 16, 75848.0, 0.3466
         UNION ALL SELECT @LATEST_SESSION_ID, 44, 75938.0, 0.4657
         UNION ALL SELECT @LATEST_SESSION_ID, 63, 76034.0, 0.5927
         UNION ALL SELECT @LATEST_SESSION_ID, 1,  76070.0, 0.6403
         UNION ALL SELECT @LATEST_SESSION_ID, 12, 76118.0, 0.7038
         UNION ALL SELECT @LATEST_SESSION_ID, 55, 76172.0, 0.7753
         UNION ALL SELECT @LATEST_SESSION_ID, 81, 76174.0, 0.7779
         UNION ALL SELECT @LATEST_SESSION_ID, 6,  76252.0, 0.8811
         UNION ALL SELECT @LATEST_SESSION_ID, 87, 76460.0, 1.1563
     ) t
WHERE t.session_id IS NOT NULL
    ON DUPLICATE KEY UPDATE
                         best_lap_ms    = VALUES(best_lap_ms),
                         qpi_pct        = VALUES(qpi_pct),
                         metric_version = VALUES(metric_version),
                         updated_at     = CURRENT_TIMESTAMP;

-- ============================================================
-- 5) starting_grid_position / forecast_quali 반영
--    @LATEST_TARGET_SESSION_ID 가 NULL이면 아무 행도 안 들어감
-- ============================================================
INSERT INTO forecast_quali (
    target_session_id,
    driver_number,
    starting_grid_position,
    exp_qpi_mean,
    exp_qpi_sigma,
    pole_prob,
    top3_prob,
    top10_prob,
    exp_grid_pos
)
SELECT t.target_session_id,
       t.driver_number,
       t.starting_grid_position,
       NULL,
       0.3000,
       t.pole_prob,
       t.top3_prob,
       t.top10_prob,
       t.exp_grid_pos
FROM (
         SELECT @LATEST_TARGET_SESSION_ID AS target_session_id, 4  AS driver_number, 1  AS starting_grid_position, 0.6464 AS pole_prob, 0.9306 AS top3_prob, 1.0000 AS top10_prob, 1.6382 AS exp_grid_pos
         UNION ALL SELECT @LATEST_TARGET_SESSION_ID, 16, 2,  0.1514, 0.6110, 0.9996, 3.3660
         UNION ALL SELECT @LATEST_TARGET_SESSION_ID, 44, 3,  0.0898, 0.4714, 0.9984, 4.0986
         UNION ALL SELECT @LATEST_TARGET_SESSION_ID, 63, 4,  0.0398, 0.2846, 0.9960, 5.1246
         UNION ALL SELECT @LATEST_TARGET_SESSION_ID, 1,  5,  0.0298, 0.2270, 0.9920, 5.5038
         UNION ALL SELECT @LATEST_TARGET_SESSION_ID, 12, 6,  0.0172, 0.1684, 0.9862, 6.0400
         UNION ALL SELECT @LATEST_TARGET_SESSION_ID, 81, 7,  0.0108, 0.1176, 0.9730, 6.5678
         UNION ALL SELECT @LATEST_TARGET_SESSION_ID, 55, 12, 0.0096, 0.1158, 0.9752, 6.5768
         UNION ALL SELECT @LATEST_TARGET_SESSION_ID, 6,  8,  0.0048, 0.0634, 0.9486, 7.4588
         UNION ALL SELECT @LATEST_TARGET_SESSION_ID, 87, 9,  0.0004, 0.0098, 0.7772, 9.3278
     ) t
WHERE t.target_session_id IS NOT NULL
    ON DUPLICATE KEY UPDATE
                         starting_grid_position = VALUES(starting_grid_position),
                         exp_qpi_sigma          = VALUES(exp_qpi_sigma),
                         pole_prob              = VALUES(pole_prob),
                         top3_prob              = VALUES(top3_prob),
                         top10_prob             = VALUES(top10_prob),
                         exp_grid_pos           = VALUES(exp_grid_pos),
                         created_at             = CURRENT_TIMESTAMP;

-- ============================================================
-- 6) fact_race_result.grid_position 동기화
--    session_id 없으면 자동으로 0건 업데이트
-- ============================================================
UPDATE fact_race_result
SET grid_position = CASE driver_number
                        WHEN 4  THEN 1
                        WHEN 16 THEN 2
                        WHEN 44 THEN 3
                        WHEN 63 THEN 4
                        WHEN 1  THEN 5
                        WHEN 12 THEN 6
                        WHEN 81 THEN 7
                        WHEN 55 THEN 12
                        WHEN 6  THEN 8
                        WHEN 87 THEN 9
                        ELSE grid_position
    END
WHERE session_id = @LATEST_SESSION_ID
  AND driver_number IN (4,16,44,63,1,12,81,55,6,87);

-- ============================================================
-- 7) VIEW 재작성
--    fr.team_name 사용 금지
--    team_name은 fact_race_result(rr)에서 조회
-- ============================================================
CREATE OR REPLACE VIEW v_latest_dashboard AS
SELECT
    fq.target_session_id AS session_id,
    fq.driver_number,
    dd.driver_name,
    dd.team_name,   -- ✅ 여기 핵심 변경

    fdm.best_lap_ms,
    fq.starting_grid_position,

    fq.exp_grid_pos,
    fq.pole_prob,
    fq.top3_prob,
    fq.top10_prob AS quali_top10_prob,

    fr.exp_points,
    fr.podium_prob,
    fr.top10_prob AS race_top10_prob,

    ast.strategy_type,
    ds.country_name,
    ds.meeting_name

FROM forecast_quali fq

         LEFT JOIN dim_driver dd
                   ON fq.driver_number = dd.driver_number

         LEFT JOIN forecast_race fr
                   ON fq.target_session_id = fr.target_session_id
                       AND fq.driver_number     = fr.driver_number

         LEFT JOIN feat_driver_session_metrics fdm
                   ON fq.target_session_id = fdm.session_id
                       AND fq.driver_number     = fdm.driver_number

         LEFT JOIN analysis_strategy ast
                   ON fq.target_session_id = ast.session_id
                       AND fq.driver_number     = ast.driver_number

         LEFT JOIN dim_session ds
                   ON fq.target_session_id = ds.session_id

WHERE fq.target_session_id = (
    SELECT MAX(target_session_id) FROM forecast_quali
);

-- ============================================================
-- 8) 검증용 조회
-- ============================================================
SELECT @LATEST_SESSION_ID AS latest_session_id,
       @LATEST_TARGET_SESSION_ID AS latest_target_session_id;

SHOW COLUMNS FROM feat_driver_session_metrics LIKE 'best_lap_ms';
SHOW COLUMNS FROM forecast_quali LIKE 'starting_grid_position';

SELECT
    fdm.session_id,
    fdm.driver_number,
    dd.driver_name,
    fdm.best_lap_ms,
    fdm.qpi_pct
FROM feat_driver_session_metrics fdm
         LEFT JOIN dim_driver dd
                   ON fdm.driver_number = dd.driver_number
WHERE fdm.session_id = @LATEST_SESSION_ID
ORDER BY fdm.best_lap_ms ASC;

SELECT
    fq.target_session_id,
    fq.driver_number,
    dd.driver_name,
    fq.starting_grid_position,
    fq.exp_grid_pos
FROM forecast_quali fq
         LEFT JOIN dim_driver dd
                   ON fq.driver_number = dd.driver_number
WHERE fq.target_session_id = @LATEST_TARGET_SESSION_ID
ORDER BY fq.exp_grid_pos ASC;

SELECT *
FROM v_latest_dashboard
         LIMIT 20;