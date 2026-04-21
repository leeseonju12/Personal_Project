package org.example.race_ready.f1web;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Controller
public class F1ResultController {

    private final String dbPath;

    public F1ResultController(@Value("${race-ready.db-path}") String dbPath) {
        this.dbPath = dbPath;
    }

    @GetMapping("/")
    public String home() {
        return "redirect:/f1";
    }

    @GetMapping("/f1")
    public String f1Result() {
        return "forward:/index.html";
    }

    @ResponseBody
    @CrossOrigin(origins = "*")
    @GetMapping("/api/dashboard")
    public Map<String, Object> dashboard() throws SQLException {
        try (Connection connection = DriverManager.getConnection("jdbc:sqlite:" + dbPath)) {
            ensureResultDashboardData(connection);

            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("session", queryOne(connection, """
                    SELECT session_id, meeting_key, year, session_name, session_type, country_name, date_start, date_end
                    FROM dim_session
                    WHERE session_id = (SELECT MAX(target_session_id) FROM forecast_race)
                    """));
            payload.put("qualifyingSummary", query(connection, """
                    SELECT
                        fdm.driver_number,
                        COALESCE(dd.driver_name, 'Driver #' || fdm.driver_number) AS driver_name,
                        COALESCE(dd.team_name, '-') AS team_name,
                        fdm.best_lap_ms,
                        fdm.qpi_pct
                    FROM feat_driver_session_metrics fdm
                    LEFT JOIN dim_driver dd
                        ON fdm.driver_number = dd.driver_number
                    LEFT JOIN fact_race_result fr
                        ON fdm.session_id = fr.session_id
                       AND fdm.driver_number = fr.driver_number
                    WHERE fdm.session_id = (SELECT MAX(session_id) FROM feat_driver_session_metrics)
                    ORDER BY fdm.best_lap_ms IS NULL, fdm.best_lap_ms ASC
                    LIMIT 10
                    """));
            payload.put("qualifyingForecast", query(connection, """
                    SELECT
                        fq.driver_number,
                        COALESCE(dd.driver_name, 'Driver #' || fq.driver_number) AS driver_name,
                        COALESCE(dd.team_name, '-') AS team_name,
                        COALESCE(fq.starting_grid_position, fr.grid_position) AS starting_grid_position,
                        fq.exp_grid_pos,
                        fq.pole_prob,
                        fq.top3_prob,
                        fq.top10_prob,
                        fq.exp_qpi_sigma AS sigma_q
                    FROM forecast_quali fq
                    LEFT JOIN dim_driver dd
                        ON fq.driver_number = dd.driver_number
                    LEFT JOIN fact_race_result fr
                        ON fq.target_session_id = fr.session_id
                       AND fq.driver_number = fr.driver_number
                    WHERE fq.target_session_id = (SELECT MAX(target_session_id) FROM forecast_quali)
                    ORDER BY fq.exp_grid_pos ASC
                    LIMIT 10
                    """));
            payload.put("raceForecast", query(connection, """
                    SELECT
                        fr.driver_number,
                        COALESCE(dd.driver_name, 'Driver #' || fr.driver_number) AS driver_name,
                        COALESCE(dd.team_name, '-') AS team_name,
                        rr.finish_position,
                        fr.exp_points,
                        fr.top10_prob,
                        fr.podium_prob
                    FROM forecast_race fr
                    LEFT JOIN dim_driver dd
                        ON fr.driver_number = dd.driver_number
                    LEFT JOIN fact_race_result rr
                        ON fr.target_session_id = rr.session_id
                       AND fr.driver_number = rr.driver_number
                    WHERE fr.target_session_id = (SELECT MAX(target_session_id) FROM forecast_race)
                    ORDER BY fr.exp_points DESC
                    LIMIT 10
                    """));
            payload.put("lapDeltaBand", lapDeltaBand(connection));
            payload.put("strategyAnalysis", query(connection, """
                    SELECT
                        ast.driver_number,
                        COALESCE(dd.driver_name, 'Driver #' || ast.driver_number) AS driver_name,
                        COALESCE(dd.team_name, '-') AS team_name,
                        fr.finish_position,
                        ast.pit_count,
                        ast.pit_median_ms,
                        ast.strategy_type,
                        ast.pit_loss_percentile,
                        ast.ir_pct,
                        ast.n_ir
                    FROM analysis_strategy ast
                    LEFT JOIN dim_driver dd
                        ON ast.driver_number = dd.driver_number
                    LEFT JOIN fact_race_result fr
                        ON ast.session_id = fr.session_id
                       AND ast.driver_number = fr.driver_number
                    WHERE ast.session_id = (SELECT MAX(session_id) FROM analysis_strategy)
                    ORDER BY ast.pit_loss_percentile DESC
                    LIMIT 10
                    """));
            return payload;
        }
    }

    private List<Map<String, Object>> lapDeltaBand(Connection connection) throws SQLException {
        List<Map<String, Object>> rows = query(connection, """
                SELECT
                    f.driver_number,
                    COALESCE(dd.driver_name, 'Driver #' || f.driver_number) AS driver_name,
                    COALESCE(dd.team_name, '-') AS team_name,
                    fr.finish_position,
                    f.rpi_pct AS mu_rpi,
                    0.2 AS sigma_rpi
                FROM feat_driver_session_metrics f
                LEFT JOIN dim_driver dd
                    ON f.driver_number = dd.driver_number
                LEFT JOIN fact_race_result fr
                    ON f.session_id = fr.session_id
                   AND f.driver_number = fr.driver_number
                WHERE f.session_id = (SELECT MAX(session_id) FROM feat_driver_session_metrics)
                  AND f.rpi_pct IS NOT NULL
                ORDER BY f.rpi_pct ASC
                LIMIT 10
                """);

        for (Map<String, Object> row : rows) {
            double mu = ((Number) row.get("mu_rpi")).doubleValue();
            double sigma = ((Number) row.get("sigma_rpi")).doubleValue();
            double bandA = normalCdf((0.3 - mu) / sigma);
            double belowBandB = normalCdf((0.7 - mu) / sigma);
            row.put("band_a_prob", bandA);
            row.put("band_b_prob", Math.max(0.0, belowBandB - bandA));
            row.put("band_c_prob", Math.max(0.0, 1.0 - belowBandB));
        }
        return rows;
    }

    private void ensureResultDashboardData(Connection connection) throws SQLException {
        try (Statement statement = connection.createStatement()) {
            statement.executeUpdate("""
                    CREATE TABLE IF NOT EXISTS dim_driver (
                        driver_number INTEGER PRIMARY KEY,
                        driver_name TEXT NOT NULL,
                        team_name TEXT
                    )
                    """);

            if (!columnExists(connection, "dim_driver", "team_name")) {
                statement.executeUpdate("ALTER TABLE dim_driver ADD COLUMN team_name TEXT");
            }
            if (!columnExists(connection, "feat_driver_session_metrics", "best_lap_ms")) {
                statement.executeUpdate("ALTER TABLE feat_driver_session_metrics ADD COLUMN best_lap_ms FLOAT");
            }
            if (!columnExists(connection, "feat_driver_session_metrics", "qpi_pct")) {
                statement.executeUpdate("ALTER TABLE feat_driver_session_metrics ADD COLUMN qpi_pct FLOAT");
            }
            if (!columnExists(connection, "forecast_quali", "starting_grid_position")) {
                statement.executeUpdate("ALTER TABLE forecast_quali ADD COLUMN starting_grid_position BIGINT");
            }

            statement.executeUpdate("""
                    INSERT INTO dim_driver (driver_number, driver_name, team_name) VALUES
                        (4,  'Lando NORRIS',     'McLaren'),
                        (16, 'Charles LECLERC',  'Ferrari'),
                        (44, 'Lewis HAMILTON',   'Ferrari'),
                        (63, 'George RUSSELL',   'Mercedes'),
                        (1,  'Max VERSTAPPEN',   'Red Bull Racing'),
                        (12, 'Kimi ANTONELLI',   'Mercedes'),
                        (55, 'Carlos SAINZ',     'Williams'),
                        (81, 'Oscar PIASTRI',    'McLaren'),
                        (6,  'Isack HADJAR',     'Racing Bulls'),
                        (87, 'Oliver BEARMAN',   'Haas F1 Team'),
                        (31, 'Esteban OCON',     'Haas F1 Team'),
                        (18, 'Lance STROLL',     'Aston Martin'),
                        (30, 'Liam LAWSON',      'Racing Bulls'),
                        (22, 'Yuki TSUNODA',     'Red Bull Racing'),
                        (14, 'Fernando ALONSO',  'Aston Martin'),
                        (10, 'Pierre GASLY',     'Alpine'),
                        (43, 'Franco COLAPINTO', 'Alpine')
                    ON CONFLICT(driver_number) DO UPDATE SET
                        driver_name = excluded.driver_name,
                        team_name = excluded.team_name
                    """);

            statement.executeUpdate("""
                    UPDATE feat_driver_session_metrics
                    SET
                        best_lap_ms = CASE driver_number
                            WHEN 4  THEN 75586.0
                            WHEN 16 THEN 75848.0
                            WHEN 44 THEN 75938.0
                            WHEN 63 THEN 76034.0
                            WHEN 1  THEN 76070.0
                            WHEN 12 THEN 76118.0
                            WHEN 55 THEN 76172.0
                            WHEN 81 THEN 76174.0
                            WHEN 6  THEN 76252.0
                            WHEN 87 THEN 76460.0
                            ELSE best_lap_ms
                        END,
                        qpi_pct = CASE driver_number
                            WHEN 4  THEN 0.0000
                            WHEN 16 THEN 0.3466
                            WHEN 44 THEN 0.4657
                            WHEN 63 THEN 0.5927
                            WHEN 1  THEN 0.6403
                            WHEN 12 THEN 0.7038
                            WHEN 55 THEN 0.7753
                            WHEN 81 THEN 0.7779
                            WHEN 6  THEN 0.8811
                            WHEN 87 THEN 1.1563
                            ELSE qpi_pct
                        END
                    WHERE session_id = (SELECT MAX(session_id) FROM feat_driver_session_metrics)
                      AND driver_number IN (4,16,44,63,1,12,55,81,6,87)
                    """);

            statement.executeUpdate("""
                    UPDATE forecast_quali
                    SET starting_grid_position = CASE driver_number
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
                        ELSE starting_grid_position
                    END
                    WHERE target_session_id = (SELECT MAX(target_session_id) FROM forecast_quali)
                      AND driver_number IN (4,16,44,63,1,12,81,55,6,87)
                    """);

            statement.executeUpdate("""
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
                    WHERE session_id = (SELECT MAX(session_id) FROM fact_race_result)
                      AND driver_number IN (4,16,44,63,1,12,81,55,6,87)
                    """);
        }
    }

    private boolean columnExists(Connection connection, String tableName, String columnName) throws SQLException {
        try (Statement statement = connection.createStatement();
             ResultSet resultSet = statement.executeQuery("PRAGMA table_info(" + tableName + ")")) {
            while (resultSet.next()) {
                if (columnName.equalsIgnoreCase(resultSet.getString("name"))) {
                    return true;
                }
            }
            return false;
        }
    }

    private Map<String, Object> queryOne(Connection connection, String sql) throws SQLException {
        List<Map<String, Object>> rows = query(connection, sql);
        return rows.isEmpty() ? Map.of() : rows.get(0);
    }

    private List<Map<String, Object>> query(Connection connection, String sql) throws SQLException {
        try (Statement statement = connection.createStatement();
             ResultSet resultSet = statement.executeQuery(sql)) {
            List<Map<String, Object>> rows = new ArrayList<>();
            int columnCount = resultSet.getMetaData().getColumnCount();
            while (resultSet.next()) {
                Map<String, Object> row = new LinkedHashMap<>();
                for (int i = 1; i <= columnCount; i++) {
                    row.put(resultSet.getMetaData().getColumnLabel(i), resultSet.getObject(i));
                }
                rows.add(row);
            }
            return rows;
        }
    }

    private double normalCdf(double x) {
        return 0.5 * (1.0 + erf(x / Math.sqrt(2.0)));
    }

    private double erf(double x) {
        double sign = Math.signum(x);
        double abs = Math.abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * abs);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t)
                + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-abs * abs);
        return sign * y;
    }
}
