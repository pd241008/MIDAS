use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct DefenseSuccessRow {
    pub attack: String,
    pub undefended: f64,
    pub adv_training: f64,
    pub input_smoothing: f64,
    pub chen_query_blinding: f64,
    pub midas_edge: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyRow {
    pub component: String,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PgdIterationRow {
    pub pgd_t: u32,
    pub defense_rate: f64,
}

pub fn write_csv<T: Serialize>(rows: &[T], path: &str) -> Result<(), String> {
    let mut wtr = csv::Writer::from_path(path)
        .map_err(|e| format!("failed to create CSV writer: {e}"))?;
    for row in rows {
        wtr.serialize(row)
            .map_err(|e| format!("failed to write CSV row: {e}"))?;
    }
    wtr.flush()
        .map_err(|e| format!("failed to flush CSV: {e}"))?;
    Ok(())
}

pub fn write_json<T: Serialize>(rows: &[T], path: &str) -> Result<(), String> {
    let json = serde_json::to_string_pretty(rows)
        .map_err(|e| format!("failed to serialize JSON: {e}"))?;
    std::fs::write(path, &json)
        .map_err(|e| format!("failed to write JSON file: {e}"))?;
    Ok(())
}

pub fn placeholder_defense_success() -> Vec<DefenseSuccessRow> {
    vec![
        DefenseSuccessRow {
            attack: "PGD T=100 eps=0.1".into(),
            undefended: 0.0,
            adv_training: 0.45,
            input_smoothing: 0.32,
            chen_query_blinding: 0.28,
            midas_edge: 0.87,
        },
        DefenseSuccessRow {
            attack: "FGSM eps=0.1".into(),
            undefended: 0.0,
            adv_training: 0.52,
            input_smoothing: 0.41,
            chen_query_blinding: 0.35,
            midas_edge: 0.92,
        },
        DefenseSuccessRow {
            attack: "C&W L2".into(),
            undefended: 0.0,
            adv_training: 0.38,
            input_smoothing: 0.29,
            chen_query_blinding: 0.31,
            midas_edge: 0.84,
        },
    ]
}

pub fn placeholder_latency() -> Vec<LatencyRow> {
    vec![
        LatencyRow {
            component: "Sensor -> Defense".into(),
            p50_ms: 0.02,
            p95_ms: 0.05,
            p99_ms: 0.08,
            max_ms: 0.12,
        },
        LatencyRow {
            component: "Momentum + Epsilon".into(),
            p50_ms: 0.01,
            p95_ms: 0.03,
            p99_ms: 0.04,
            max_ms: 0.06,
        },
        LatencyRow {
            component: "Givens Rotation".into(),
            p50_ms: 0.08,
            p95_ms: 0.15,
            p99_ms: 0.22,
            max_ms: 0.35,
        },
        LatencyRow {
            component: "Inference (Mock)".into(),
            p50_ms: 0.05,
            p95_ms: 0.05,
            p99_ms: 0.06,
            max_ms: 0.07,
        },
    ]
}

pub fn placeholder_pgd_iterations() -> Vec<PgdIterationRow> {
    vec![
        PgdIterationRow {
            pgd_t: 20,
            defense_rate: 0.91,
        },
        PgdIterationRow {
            pgd_t: 50,
            defense_rate: 0.88,
        },
        PgdIterationRow {
            pgd_t: 100,
            defense_rate: 0.87,
        },
    ]
}
