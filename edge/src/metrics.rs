use serde::Serialize;
use std::time::Duration;

use crate::channels::Phase;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PhaseLatencySample {
    pub phase: Phase,
    pub latency: Duration,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhaseLatencySummary {
    pub archimedean: LatencyHistogram,
    pub logarithmic: LatencyHistogram,
}

impl PhaseLatencySummary {
    pub fn from_samples(samples: &[PhaseLatencySample]) -> Self {
        let mut arch = Vec::new();
        let mut log = Vec::new();
        for s in samples {
            match s.phase {
                Phase::Archimedean => arch.push(s.latency),
                Phase::Logarithmic => log.push(s.latency),
            }
        }
        Self {
            archimedean: LatencyHistogram::from_samples(arch),
            logarithmic: LatencyHistogram::from_samples(log),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyHistogram {
    pub samples: Vec<f64>,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub count: usize,
}

impl LatencyHistogram {
    pub fn from_samples(samples: Vec<Duration>) -> Self {
        let count = samples.len();
        if count == 0 {
            return Self {
                samples: vec![],
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                max_ms: 0.0,
                count: 0,
            };
        }

        let mut ms: Vec<f64> = samples
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&ms, 50.0);
        let p95 = percentile(&ms, 95.0);
        let p99 = percentile(&ms, 99.0);
        let max = ms[count - 1];

        Self {
            samples: ms,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
            max_ms: max,
            count,
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[derive(Debug, Clone)]
pub struct CpuSample {
    pub timestamp_ns: u128,
    pub cpu_percent: f64,
}

pub struct CpuMonitor {
    pid: u32,
    prev_total: u64,
    prev_idle: u64,
    prev_process: u64,
}

impl CpuMonitor {
    pub fn new() -> Self {
        let pid = std::process::id();
        let (total, idle) = read_cpu_times().unwrap_or((0, 0));
        let process = read_process_cpu(pid).unwrap_or(0);
        Self {
            pid,
            prev_total: total,
            prev_idle: idle,
            prev_process: process,
        }
    }

    pub fn sample(&mut self) -> Option<f64> {
        let (total, idle) = read_cpu_times()?;
        let process = read_process_cpu(self.pid)?;

        let total_delta = total.saturating_sub(self.prev_total);
        let process_delta = process.saturating_sub(self.prev_process);

        self.prev_total = total;
        self.prev_idle = idle;
        self.prev_process = process;

        if total_delta == 0 {
            return None;
        }

        let cpu_pct = (process_delta as f64 / total_delta as f64) * 100.0;
        Some(cpu_pct)
    }
}

fn read_cpu_times() -> Option<(u64, u64)> {
    let content = std::fs::read_to_string("/proc/stat").ok()?;
    let line = content.lines().next()?;
    let vals: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    if vals.len() < 5 {
        return None;
    }
    let total: u64 = vals.iter().sum();
    let idle = vals[3];
    Some((total, idle))
}

fn read_process_cpu(pid: u32) -> Option<u64> {
    let content = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let fields: Vec<&str> = content.split_whitespace().collect();
    if fields.len() < 15 {
        return None;
    }
    let utime: u64 = fields.get(13).and_then(|s| s.parse().ok()).unwrap_or(0);
    let stime: u64 = fields.get(14).and_then(|s| s.parse().ok()).unwrap_or(0);
    Some(utime + stime)
}
