import * as React from "react"

/**
 * Framer Code Component (React/TSX)
 * - Paste into a new Code Component
 * - Drag onto canvas
 * - Loads dashboard data from the Spring API at /api/dashboard
 */

type Row = Record<string, any>
type DashboardPayload = {
    qualifyingSummary?: Row[]
    qualifyingForecast?: Row[]
    raceForecast?: Row[]
    lapDeltaBand?: Row[]
    strategyAnalysis?: Row[]
}

const ACCENT = "#E10600"
const BG = "#000000"
const CARD = "rgba(255,255,255,0.06)"
const BORDER = "rgba(255,255,255,0.10)"
const TEXT = "rgba(255,255,255,0.92)"
const MUTED = "rgba(255,255,255,0.65)"

const fmtPct = (x: number) => `${(x * 100).toFixed(2)}%`
const fmtNum = (x: number | null | undefined, digits: number) =>
    x == null || Number.isNaN(Number(x)) ? "-" : Number(x).toFixed(digits)
const fmtInt = (x: number | null | undefined) =>
    x == null || Number.isNaN(Number(x)) ? "-" : String(x)
const toNumber = (x: unknown, fallback = 0) => {
    const n = Number(x)
    return Number.isFinite(n) ? n : fallback
}
const clamp01 = (x: unknown) => Math.max(0, Math.min(1, toNumber(x)))

function msToTime(ms: number) {
    if (ms == null || isNaN(ms)) return "-"
    const total = Math.round(ms)
    const m = Math.floor(total / 60000)
    const s = Math.floor((total % 60000) / 1000)
    const mm = total % 1000
    return `${m}:${String(s).padStart(2, "0")}.${String(mm).padStart(3, "0")}`
}

function Pill({
                  children,
                  tone = "muted",
              }: {
    children: React.ReactNode
    tone?: "muted" | "accent"
}) {
    const bg =
        tone === "accent" ? "rgba(225,6,0,0.14)" : "rgba(255,255,255,0.08)"
    const bd = tone === "accent" ? "rgba(225,6,0,0.35)" : BORDER
    const color = tone === "accent" ? "rgba(255,255,255,0.92)" : MUTED
    return (
        <span
            style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 10px",
                borderRadius: 999,
                background: bg,
                border: `1px solid ${bd}`,
                color,
                fontSize: 12,
                lineHeight: "12px",
                whiteSpace: "nowrap",
            }}
        >
            {children}
        </span>
    )
}

function ProgressCell({
                          value,
                          color = ACCENT,
                      }: {
    value: unknown
    color?: string
}) {
    const v = clamp01(value)
    return (
        <div style={{ display: "grid", gap: 6 }}>
            <div
                style={{
                    height: 8,
                    borderRadius: 999,
                    background: "rgba(255,255,255,0.10)",
                    overflow: "hidden",
                    border: `1px solid rgba(255,255,255,0.08)`,
                }}
                aria-label={`progress ${fmtPct(v)}`}
            >
                <div
                    style={{
                        width: `${v * 100}%`,
                        height: "100%",
                        background: color,
                    }}
                />
            </div>
            <div style={{ fontSize: 12, color: MUTED }}>{fmtPct(v)}</div>
        </div>
    )
}

function Donut({ value, label }: { value: unknown; label: string }) {
    const v = clamp01(value)
    const size = 22
    return (
        <span
            style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                whiteSpace: "nowrap",
            }}
        >
            <span
                style={{
                    width: size,
                    height: size,
                    borderRadius: "50%",
                    background: `conic-gradient(${ACCENT} ${v * 360}deg, rgba(255,255,255,0.12) 0deg)`,
                    border: `1px solid rgba(255,255,255,0.12)`,
                }}
                aria-label={`${label} ${fmtPct(v)}`}
                title={`${label} ${fmtPct(v)}`}
            />
            <span style={{ fontSize: 12, color: MUTED }}>{fmtPct(v)}</span>
        </span>
    )
}

function Section({
                     title,
                     subtitle,
                     koDesc,
                     children,
                 }: {
    title: string
    subtitle?: string
    koDesc?: string
    children: React.ReactNode
}) {
    return (
        <section style={{ display: "grid", gap: 14 }}>
            <div style={{ display: "grid", gap: 6 }}>
                <div
                    style={{
                        fontSize: 18,
                        fontWeight: 800,
                        color: TEXT,
                        letterSpacing: -0.2,
                    }}
                >
                    {title}
                </div>
                {subtitle ? (
                    <div style={{ fontSize: 13, color: MUTED }}>{subtitle}</div>
                ) : null}
                {koDesc ? (
                    <div style={{ fontSize: 13, color: MUTED }}>{koDesc}</div>
                ) : null}
            </div>
            <div
                style={{
                    background: CARD,
                    border: `1px solid ${BORDER}`,
                    borderRadius: 12,
                    padding: 14,
                    overflow: "hidden",
                    boxShadow:
                        "0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06)",
                    backdropFilter: "blur(10px)",
                }}
            >
                {children}
            </div>
        </section>
    )
}

function Table({
                   columns,
                   rows,
                   rowKey,
                   highlightRowIndex,
                   mobileCard,
               }: {
    columns: Array<{
        key: string
        title: string
        align?: "left" | "right" | "center"
        render?: (row: Row, idx: number) => React.ReactNode
        width?: string | number
    }>
    rows: Row[]
    rowKey: (row: Row, idx: number) => string
    highlightRowIndex?: number
    mobileCard?: (row: Row, idx: number) => React.ReactNode
}) {
    return (
        <>
            <div className="rr-tableWrap">
                <table className="rr-table">
                    <thead>
                    <tr>
                        {columns.map((c) => (
                            <th
                                key={c.key}
                                style={{
                                    textAlign: c.align ?? "left",
                                    width: c.width,
                                }}
                            >
                                {c.title}
                            </th>
                        ))}
                    </tr>
                    </thead>
                    <tbody>
                    {rows.map((r, idx) => {
                        const isHighlight = idx === highlightRowIndex
                        return (
                            <tr
                                key={rowKey(r, idx)}
                                className={
                                    isHighlight ? "rr-highlight" : ""
                                }
                            >
                                {columns.map((c) => (
                                    <td
                                        key={c.key}
                                        style={{
                                            textAlign: c.align ?? "left",
                                        }}
                                    >
                                        {c.render
                                            ? c.render(r, idx)
                                            : r[c.key]}
                                    </td>
                                ))}
                            </tr>
                        )
                    })}
                    </tbody>
                </table>
            </div>

            <div className="rr-cards">
                {rows.map((r, idx) => {
                    const isHighlight = idx === highlightRowIndex
                    return (
                        <div
                            key={rowKey(r, idx)}
                            className={`rr-card ${isHighlight ? "rr-cardHighlight" : ""}`}
                        >
                            {mobileCard ? (
                                mobileCard(r, idx)
                            ) : (
                                <div style={{ display: "grid", gap: 10 }}>
                                    {columns.map((c) => (
                                        <div
                                            key={c.key}
                                            style={{
                                                display: "flex",
                                                justifyContent: "space-between",
                                                gap: 12,
                                            }}
                                        >
                                            <div
                                                style={{
                                                    color: MUTED,
                                                    fontSize: 12,
                                                }}
                                            >
                                                {c.title}
                                            </div>
                                            <div
                                                style={{
                                                    color: TEXT,
                                                    fontSize: 13,
                                                }}
                                            >
                                                {c.render
                                                    ? c.render(r, idx)
                                                    : r[c.key]}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>
        </>
    )
}

function StackedBands({ a, b, c }: { a: unknown; b: unknown; c: unknown }) {
    const A = clamp01(a)
    const B = clamp01(b)
    const C = clamp01(c)
    const total = A + B + C || 1
    const aW = (A / total) * 100
    const bW = (B / total) * 100
    const cW = (C / total) * 100
    return (
        <div style={{ display: "grid", gap: 8 }}>
            <div
                style={{
                    height: 10,
                    borderRadius: 999,
                    overflow: "hidden",
                    background: "rgba(255,255,255,0.10)",
                    border: `1px solid rgba(255,255,255,0.08)`,
                    display: "flex",
                }}
                aria-label={`Band A ${fmtPct(A)}, Band B ${fmtPct(B)}, Band C ${fmtPct(C)}`}
            >
                <div style={{ width: `${aW}%`, background: ACCENT }} />
                <div
                    style={{
                        width: `${bW}%`,
                        background: "rgba(255,255,255,0.35)",
                    }}
                />
                <div
                    style={{
                        width: `${cW}%`,
                        background: "rgba(255,255,255,0.18)",
                    }}
                />
            </div>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <Pill tone="accent">Band A: {fmtPct(A)}</Pill>
                <Pill>Band B: {fmtPct(B)}</Pill>
                <Pill>Band C: {fmtPct(C)}</Pill>
            </div>
        </div>
    )
}

function DriverTag({ row }: { row: any }) {
    return (
        <div style={{ display: "grid", gap: 2 }}>
            <div style={{ fontWeight: 850, letterSpacing: 0.2 }}>
                {row.driver_name}{" "}
                <span style={{ color: MUTED, fontWeight: 700 }}>
                    #{row.driver_number}
                </span>
            </div>
            <div style={{ fontSize: 12, color: MUTED }}>{row.team_name}</div>
        </div>
    )
}

export default function RaceReadyDashboard() {
    const [dashboard, setDashboard] = React.useState<DashboardPayload | null>(null)
    const [loading, setLoading] = React.useState(true)
    const [error, setError] = React.useState<string | null>(null)

    React.useEffect(() => {
        let ignore = false

        async function loadDashboard() {
            try {
                setLoading(true)
                setError(null)
                const response = await fetch("http://localhost:8080/api/dashboard", {
                    headers: { Accept: "application/json" },
                })
                if (!response.ok) {
                    throw new Error(`Dashboard API failed: ${response.status}`)
                }
                const payload = await response.json()
                if (!ignore) setDashboard(payload)
            } catch (err) {
                if (!ignore) {
                    setError(err instanceof Error ? err.message : String(err))
                }
            } finally {
                if (!ignore) setLoading(false)
            }
        }

        loadDashboard()
        return () => {
            ignore = true
        }
    }, [])

    const qualifyingSummary = dashboard?.qualifyingSummary ?? []
    const qualifyingForecast = dashboard?.qualifyingForecast ?? []
    const raceForecast = dashboard?.raceForecast ?? []
    const lapDeltaBand = dashboard?.lapDeltaBand ?? []
    const strategyAnalysis = dashboard?.strategyAnalysis ?? []

    const fastestPitIdx = strategyAnalysis.reduce((bestIdx: number, row: Row, idx: number, arr: Row[]) => {
        const cur = row.pit_median_ms == null ? null : toNumber(row.pit_median_ms, Number.POSITIVE_INFINITY)
        const best = arr[bestIdx]?.pit_median_ms == null ? null : toNumber(arr[bestIdx]?.pit_median_ms, Number.POSITIVE_INFINITY)
        if (best == null) return idx
        if (cur == null) return bestIdx
        return cur < best ? idx : bestIdx
    }, 0)
    return (
        <div
            style={{
                width: "100%",
                height: "100%",
                background: BG,
                color: TEXT,
                fontFamily:
                    'Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji"',
                overflow: "auto",
            }}
        >
            <style>{`
                .rr-container{
                    max-width: 1100px;
                    margin: 0 auto;
                    padding: 28px 18px 48px;
                    display: grid;
                    gap: 28px;
                }
                .rr-hero{ display:grid; gap:10px; }
                .rr-title{
                    font-size: 34px; line-height:1.05; letter-spacing:-0.6px;
                    font-weight: 850; margin:0;
                }
                .rr-sub{ font-size:14px; color:${MUTED}; margin:0; }
                .rr-divider{
                    height: 2px; width:72px; background:${ACCENT};
                    border-radius:999px; margin-top:10px;
                }
                .rr-grid{ display:grid; gap:28px; }
                .rr-tableWrap{ display:block; }
                .rr-table{ width:100%; border-collapse:collapse; }
                .rr-table th{
                    font-size:12px; color:${MUTED}; font-weight:650;
                    text-transform:uppercase; letter-spacing:0.08em;
                    padding:10px 10px; border-bottom:1px solid rgba(255,255,255,0.10);
                }
                .rr-table td{
                    padding:12px 10px; border-bottom:1px solid rgba(255,255,255,0.08);
                    font-size:13px; color:${TEXT}; vertical-align:middle;
                }
                .rr-table tbody tr:hover td{ background: rgba(255,255,255,0.04); }
                .rr-highlight td{
                    background: rgba(225,6,0,0.10);
                    box-shadow: inset 0 0 0 1px rgba(225,6,0,0.22);
                }
                .rr-cards{ display:none; gap:12px; }
                .rr-card{
                    background: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 12px;
                    padding: 12px;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
                }
                .rr-cardHighlight{
                    background: rgba(225,6,0,0.10);
                    border-color: rgba(225,6,0,0.30);
                }
                @media (max-width: 760px){
                    .rr-title{ font-size: 28px; }
                    .rr-tableWrap{ display:none; }
                    .rr-cards{ display:grid; }
                }
            `}</style>

            <div className="rr-container">
                <div className="rr-hero">
                    <h1 className="rr-title">
                        Grand Prix AI Performance Analysis
                    </h1>
                    <p className="rr-sub">
                        Qualifying / Forecast / Race Simulation / Strategy
                    </p>
                    <div className="rr-divider" />
                </div>

                {loading || error ? (
                    <div
                        style={{
                            background: CARD,
                            border: `1px solid ${error ? "rgba(225,6,0,0.55)" : BORDER}`,
                            borderRadius: 12,
                            padding: 14,
                            color: error ? TEXT : MUTED,
                            fontSize: 13,
                        }}
                    >
                        {error ? `Dashboard API error: ${error}` : "Loading dashboard data..."}
                    </div>
                ) : null}

                <div className="rr-grid">
                    {/* 1) Qualifying Summary */}
                    <Section
                        title="Qualifying Summary - Top 10"
                        subtitle="Best lap, QPI, and teammate delta."
                        koDesc="DB에 저장된 예선 성능 지표를 기준으로 상위 10명을 표시합니다."
                    >
                        <Table
                            columns={[
                                {
                                    key: "driver",
                                    title: "Driver",
                                    render: (r) => <DriverTag row={r} />,
                                    width: "38%",
                                },
                                {
                                    key: "best_lap_ms",
                                    title: "Best Lap",
                                    align: "right",
                                    render: (r) => (
                                        <span
                                            style={{
                                                fontVariantNumeric:
                                                    "tabular-nums",
                                            }}
                                        >
                                            {msToTime(r.best_lap_ms)}
                                        </span>
                                    ),
                                },
                                {
                                    key: "qpi_pct",
                                    title: "QPI",
                                    align: "right",
                                    render: (r) => (
                                        <span
                                            style={{
                                                fontVariantNumeric:
                                                    "tabular-nums",
                                            }}
                                        >
                                            {fmtNum(r.qpi_pct, 6)}
                                        </span>
                                    ),
                                },
                            ]}
                            rows={qualifyingSummary}
                            rowKey={(r) => `qs-${r.driver_number}`}
                            highlightRowIndex={0}
                        />
                    </Section>

                    {/* 2) Qualifying Forecast */}
                    <Section
                        title="Qualifying Forecast - Top 10"
                        subtitle="Probabilities and expected grid position."
                        koDesc="폴 포지션, Top 3, Top 10 확률과 예상 그리드 순위를 제공합니다."
                    >
                        <Table
                            columns={[
                                {
                                    key: "driver",
                                    title: "Driver",
                                    render: (r) => <DriverTag row={r} />,
                                    width: "34%",
                                },
                                {
                                    key: "starting_grid_position",
                                    title: "Grid",
                                    align: "right",
                                    render: (r) => (
                                        <span
                                            style={{
                                                fontVariantNumeric:
                                                    "tabular-nums",
                                            }}
                                        >
                                            {fmtInt(r.starting_grid_position)}
                                        </span>
                                    ),
                                },
                                {
                                    key: "exp_grid_pos",
                                    title: "Exp Grid",
                                    align: "right",
                                    render: (r) => (
                                        <span
                                            style={{
                                                fontVariantNumeric:
                                                    "tabular-nums",
                                            }}
                                        >
                                            {fmtNum(r.exp_grid_pos, 4)}
                                        </span>
                                    ),
                                },
                                {
                                    key: "pole_prob",
                                    title: "Pole",
                                    render: (r) => (
                                        <ProgressCell
                                            value={r.pole_prob}
                                            color={ACCENT}
                                        />
                                    ),
                                },
                                {
                                    key: "top3_prob",
                                    title: "Top 3",
                                    render: (r) => (
                                        <ProgressCell
                                            value={r.top3_prob}
                                            color={"rgba(225,6,0,0.55)"}
                                        />
                                    ),
                                },
                                {
                                    key: "top10_prob",
                                    title: "Top 10",
                                    render: (r) => (
                                        <ProgressCell
                                            value={r.top10_prob}
                                            color={"rgba(255,255,255,0.40)"}
                                        />
                                    ),
                                },
                            ]}
                            rows={qualifyingForecast}
                            rowKey={(r) => `qf-${r.driver_number}`}
                            highlightRowIndex={0}
                        />
                    </Section>

                    {/* 3) Race Forecast */}
                    <Section
                        title="Race Forecast - Top 10"
                        subtitle="Expected points and outcome probabilities."
                        koDesc="레이스 시뮬레이션 기반 기대 포인트와 Top 10/포디움 확률을 표시합니다."
                    >
                        <Table
                            columns={[
                                {
                                    key: "driver",
                                    title: "Driver",
                                    render: (r) => <DriverTag row={r} />,
                                    width: "34%",
                                },
                                {
                                    key: "finish_position",
                                    title: "Finish",
                                    align: "right",
                                    render: (r) => (
                                        <span
                                            style={{
                                                fontVariantNumeric:
                                                    "tabular-nums",
                                            }}
                                        >
                                            {fmtInt(r.finish_position)}
                                        </span>
                                    ),
                                },
                                {
                                    key: "exp_points",
                                    title: "Exp Pts",
                                    align: "right",
                                    render: (r) => (
                                        <span
                                            style={{
                                                fontVariantNumeric:
                                                    "tabular-nums",
                                                fontWeight: 800,
                                            }}
                                        >
                                            {fmtNum(r.exp_points, 6)}
                                        </span>
                                    ),
                                },
                                {
                                    key: "top10_prob",
                                    title: "Top 10",
                                    render: (r) => (
                                        <Donut
                                            value={r.top10_prob}
                                            label="Top10"
                                        />
                                    ),
                                },
                                {
                                    key: "podium_prob",
                                    title: "Podium",
                                    render: (r) => (
                                        <Donut
                                            value={r.podium_prob}
                                            label="Podium"
                                        />
                                    ),
                                },
                            ]}
                            rows={[...raceForecast].sort(
                                (a, b) => b.exp_points - a.exp_points
                            )}
                            rowKey={(r) => `rf-${r.driver_number}`}
                            highlightRowIndex={0}
                        />
                    </Section>

                    {/* 4) Lap Delta Band */}
                    <Section
                        title="Lap Delta Band - Top 10"
                        subtitle="Pace bands with mean and sigma RPI."
                        koDesc="랩 페이스 분포를 Band A/B/C로 시각화하고 RPI 평균과 변동성을 함께 보여줍니다."
                    >
                        <Table
                            columns={[
                                {
                                    key: "driver",
                                    title: "Driver",
                                    render: (r) => <DriverTag row={r} />,
                                    width: "30%",
                                },
                                {
                                    key: "bands",
                                    title: "Pace Bands",
                                    render: (r) => (
                                        <StackedBands
                                            a={r.band_a_prob}
                                            b={r.band_b_prob}
                                            c={r.band_c_prob}
                                        />
                                    ),
                                    width: "40%",
                                },
                                {
                                    key: "mu_rpi",
                                    title: "Mean RPI",
                                    align: "right",
                                    render: (r) => (
                                        <span
                                            style={{
                                                fontVariantNumeric:
                                                    "tabular-nums",
                                            }}
                                        >
                                            {fmtNum(r.mu_rpi, 6)}
                                        </span>
                                    ),
                                },
                                {
                                    key: "sigma_rpi",
                                    title: "Sigma RPI",
                                    align: "right",
                                    render: (r) => (
                                        <span
                                            style={{
                                                fontVariantNumeric:
                                                    "tabular-nums",
                                            }}
                                        >
                                            {fmtNum(r.sigma_rpi, 1)}
                                        </span>
                                    ),
                                },
                            ]}
                            rows={lapDeltaBand}
                            rowKey={(r) => `ldb-${r.driver_number}`}
                            highlightRowIndex={0}
                        />
                    </Section>

                    {/* 5) Strategy Analysis */}
                    <Section
                        title="Strategy Analysis - Top 10"
                        subtitle="Pit stops, median pit time, and loss percentile."
                        koDesc="피트스톱 횟수, 중앙값, 손실 분위수를 기준으로 전략 효율을 비교합니다."
                    >
                        <Table
                            columns={[
                                {
                                    key: "driver",
                                    title: "Driver",
                                    render: (r) => <DriverTag row={r} />,
                                    width: "34%",
                                },
                                {
                                    key: "pit_count",
                                    title: "Pit",
                                    align: "right",
                                },
                                {
                                    key: "pit_median_ms",
                                    title: "Median Pit",
                                    align: "right",
                                    render: (r) => (
                                        <span
                                            style={{
                                                fontVariantNumeric:
                                                    "tabular-nums",
                                            }}
                                        >
                                            {fmtNum(r.pit_median_ms, 1)} ms
                                        </span>
                                    ),
                                },
                                {
                                    key: "strategy_type",
                                    title: "Strategy",
                                    render: (r) => (
                                        <Pill
                                            tone={
                                                r.strategy_type === "2-stop"
                                                    ? "accent"
                                                    : "muted"
                                            }
                                        >
                                            {r.strategy_type}
                                        </Pill>
                                    ),
                                },
                                {
                                    key: "pit_loss_percentile",
                                    title: "Pit Loss %ile",
                                    align: "right",
                                    render: (r) => (
                                        <span
                                            style={{
                                                fontVariantNumeric:
                                                    "tabular-nums",
                                            }}
                                        >
                                            {fmtNum(r.pit_loss_percentile, 6)}
                                        </span>
                                    ),
                                },
                            ]}
                            rows={strategyAnalysis}
                            rowKey={(r) => `sa-${r.driver_number}`}
                            highlightRowIndex={fastestPitIdx}
                        />
                    </Section>
                </div>
            </div>
        </div>
    )
}
