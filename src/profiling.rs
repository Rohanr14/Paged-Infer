//! Optional diagnostic stage timers. Normal builds contain no clocks/counters.
//!
//! Enable with `--features profiling`. Counters are process-global: reset and
//! snapshot only between inference calls with all workers joined. Concurrent
//! engines otherwise contribute to the same totals. Timings include scheduling
//! delays; parallel lane times are summed and overlap the main attention span.
//! Instrumented runs must not be used to decide performance gates.

use serde::Serialize;

#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Stage {
    ModelSetup,
    Elementwise,
    QkvProjection,
    RopeKvWrite,
    Attention,
    OutputProjection,
    FeedForwardProjection,
    LmHead,
    SharedSetup,
    SharedPack,
    SharedScores,
    PrivateScores,
    Softmax,
    SharedValues,
    PrivateValues,
    SharedScatter,
}

/// Accumulated elapsed time of stage spans, not CPU time or exclusive wall time.
#[derive(Debug, Serialize)]
pub struct StageTiming {
    pub stage: &'static str,
    pub scope: &'static str,
    pub calls: u64,
    pub elapsed_ms: f64,
}

pub const fn enabled() -> bool {
    cfg!(feature = "profiling")
}

pub struct Span {
    #[cfg(feature = "profiling")]
    stage: Stage,
    #[cfg(feature = "profiling")]
    started: std::time::Instant,
}

impl Span {
    #[inline]
    pub fn new(stage: Stage) -> Self {
        #[cfg(not(feature = "profiling"))]
        let _ = stage;
        Self {
            #[cfg(feature = "profiling")]
            stage,
            #[cfg(feature = "profiling")]
            started: std::time::Instant::now(),
        }
    }

    /// Finish the current span and begin the next; a no-op in normal builds.
    #[inline]
    pub fn enter(&mut self, stage: Stage) {
        #[cfg(feature = "profiling")]
        {
            self.record();
            self.stage = stage;
            self.started = std::time::Instant::now();
        }
        #[cfg(not(feature = "profiling"))]
        let _ = stage;
    }

    #[cfg(feature = "profiling")]
    fn record(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let ns = self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        counters::NANOSECONDS[self.stage as usize].fetch_add(ns, Relaxed);
        counters::CALLS[self.stage as usize].fetch_add(1, Relaxed);
    }
}

impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        #[cfg(feature = "profiling")]
        self.record();
    }
}

/// Clear all counters, after inference has joined its workers.
pub fn reset() {
    #[cfg(feature = "profiling")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        for counter in counters::NANOSECONDS.iter().chain(&counters::CALLS) {
            counter.store(0, Relaxed);
        }
    }
}

/// Main-thread and worker totals overlap and must not be added together.
pub fn snapshot() -> Vec<StageTiming> {
    #[cfg(feature = "profiling")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        counters::NAMES
            .iter()
            .enumerate()
            .map(|(index, &(stage, scope))| StageTiming {
                stage,
                scope,
                calls: counters::CALLS[index].load(Relaxed),
                elapsed_ms: counters::NANOSECONDS[index].load(Relaxed) as f64 / 1_000_000.0,
            })
            .collect()
    }
    #[cfg(not(feature = "profiling"))]
    Vec::new()
}

#[cfg(feature = "profiling")]
mod counters {
    use std::sync::atomic::AtomicU64;
    pub const NAMES: [(&str, &str); 16] = [
        ("model_setup", "model_main"),
        ("elementwise", "model_main"),
        ("qkv_projection", "model_main"),
        ("rope_kv_write", "model_main"),
        ("attention", "model_main"),
        ("output_projection", "model_main"),
        ("feed_forward_projection", "model_main"),
        ("lm_head", "model_main"),
        ("shared_setup", "attention_main"),
        ("shared_pack", "attention_workers"),
        ("shared_scores", "attention_workers"),
        ("private_scores", "attention_workers"),
        ("softmax", "attention_workers"),
        ("shared_values", "attention_workers"),
        ("private_values", "attention_workers"),
        ("shared_scatter", "attention_main"),
    ];
    pub static NANOSECONDS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
    pub static CALLS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
}
