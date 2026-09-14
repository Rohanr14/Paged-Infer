//! Reproducible engine workloads and measurements at the streaming consumer boundary.
//!
//! Millisecond arrivals are open-loop: a blocking engine step does not move their
//! deadlines. Step arrivals give deterministic dispatch for a fixed scheduler.
//! Both modes measure real elapsed time; only millisecond mode includes a planned
//! arrival's dispatch delay in latency. This driver does not measure HTTP latency.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::engine::{Completion, Engine, FinishReason, RequestOptions};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayClock {
    Steps,
    Milliseconds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub version: u32,
    pub clock: ReplayClock,
    pub events: Vec<ReplayEvent>,
}

fn one_sample() -> usize {
    1
}
fn unit_top_p() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReplayEvent {
    Submit {
        at: u64,
        id: String,
        tokens: Vec<u32>,
        max_tokens: usize,
        #[serde(default = "one_sample")]
        samples: usize,
        /// Required: replay must not depend on global request/sequence IDs.
        seed: u64,
        /// Explicit zero also keeps forked samples greedy.
        #[serde(default)]
        temperature: f32,
        #[serde(default = "unit_top_p")]
        top_p: f32,
        #[serde(default)]
        top_k: usize,
    },
    Cancel {
        at: u64,
        id: String,
    },
}

impl ReplayEvent {
    pub fn at(&self) -> u64 {
        match self {
            Self::Submit { at, .. } | Self::Cancel { at, .. } => *at,
        }
    }
}

impl Workload {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == 1,
            "unsupported workload version {}",
            self.version
        );
        ensure!(!self.events.is_empty(), "workload must contain events");
        let mut ids = HashSet::new();
        let mut cancelled = HashSet::new();
        let mut previous = 0;
        for (index, event) in self.events.iter().enumerate() {
            ensure!(event.at() >= previous, "event {index} is out of order");
            previous = event.at();
            match event {
                ReplayEvent::Submit {
                    id,
                    tokens,
                    max_tokens,
                    samples,
                    temperature,
                    top_p,
                    ..
                } => {
                    ensure!(
                        !id.is_empty() && ids.insert(id),
                        "empty or duplicate request id {id:?}"
                    );
                    ensure!(!tokens.is_empty(), "request {id}: empty prompt");
                    ensure!(
                        *max_tokens > 0 && *samples > 0,
                        "request {id}: budget and samples must be positive"
                    );
                    ensure!(
                        tokens.len().checked_add(*max_tokens).is_some()
                            && tokens.len().checked_add(*samples).is_some(),
                        "request {id}: token or sample count overflows"
                    );
                    ensure!(
                        temperature.is_finite() && *temperature >= 0.0,
                        "request {id}: temperature must be finite and nonnegative"
                    );
                    ensure!(
                        top_p.is_finite() && *top_p > 0.0 && *top_p <= 1.0,
                        "request {id}: top_p must be in (0, 1]"
                    );
                }
                ReplayEvent::Cancel { id, .. } => {
                    ensure!(
                        ids.contains(id),
                        "cancel references request {id:?} before submission"
                    );
                    ensure!(cancelled.insert(id), "duplicate cancel for request {id:?}");
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ReplayLimits {
    pub max_steps: usize,
    pub timeout: Duration,
}

impl Default for ReplayLimits {
    fn default() -> Self {
        Self {
            max_steps: 100_000,
            timeout: Duration::from_secs(60),
        }
    }
}

/// Nearest-rank percentiles. An empty population is null, never a zero latency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution {
    pub count: usize,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub max: Option<f64>,
}

impl Distribution {
    pub fn from_samples(mut samples: Vec<f64>) -> Self {
        samples.sort_by(f64::total_cmp);
        let percentile = |percent: usize| {
            if samples.is_empty() {
                None
            } else {
                Some(samples[(samples.len() * percent).div_ceil(100) - 1])
            }
        };
        Self {
            count: samples.len(),
            p50: percentile(50),
            p95: percentile(95),
            p99: percentile(99),
            max: samples.last().copied(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    Finished,
    Rejected,
    CancelledBeforeAdmission,
}

/// A batch observable when the driver drains the engine after a step/cancel.
/// Every token in this batch has the same timestamp. Counts partition `tokens`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delivery {
    pub at_ms: f64,
    pub tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceRecord {
    pub sample_index: usize,
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    pub deliveries: Vec<Delivery>,
    pub ttft_ms: Option<f64>,
    pub engine_ttft_ms: Option<f64>,
    pub engine_queue_ms: f64,
    pub end_to_end_ms: f64,
    pub finished_at_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRecord {
    pub id: String,
    pub scheduled_at: u64,
    pub submitted_at_ms: f64,
    /// Null for the step clock: a step index has no wall-time deadline.
    pub dispatch_lag_ms: Option<f64>,
    pub cancel_scheduled_at: Option<u64>,
    pub cancel_dispatched_at_ms: Option<f64>,
    pub cancel_dispatch_lag_ms: Option<f64>,
    pub status: RequestStatus,
    pub rejection: Option<String>,
    pub finished_at_ms: f64,
    pub sequences: Vec<SequenceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySummary {
    pub requests: usize,
    pub successful_requests: usize,
    pub rejected_requests: usize,
    pub cancelled_requests: usize,
    pub failed_requests: usize,
    pub completed_sequences: usize,
    pub cancelled_sequences: usize,
    pub oom_sequences: usize,
    pub output_tokens: usize,
    pub useful_output_tokens: usize,
    pub completed_requests_per_second: f64,
    pub useful_tokens_per_second: f64,
    pub dispatch_lag_ms: Distribution,
    pub ttft_ms: Distribution,
    pub engine_queue_ms: Distribution,
    pub end_to_end_ms: Distribution,
    /// Observed token gaps, including zero gaps within a delivery batch.
    pub observed_inter_token_ms: Distribution,
    /// Gaps between nonempty delivery batches; no invented within-batch timing.
    pub inter_delivery_ms: Distribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCounters {
    pub steps: usize,
    pub admitted_requests: usize,
    pub prompt_tokens: usize,
    pub prompt_tokens_prefilled: usize,
    /// Explicit cache skips; unfinished/cancelled positions are not reuse.
    #[serde(default)]
    pub prompt_tokens_reused: usize,
    pub generated_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub preemptions: usize,
    #[serde(default)]
    pub prefill_preemptions: usize,
    #[serde(default)]
    pub prefill_chunks: usize,
    #[serde(default)]
    pub last_prefill_tokens: usize,
    pub recomputed_tokens: usize,
    pub deferred_steps: usize,
    pub prefix_hits: u64,
    pub prefix_lookups: u64,
    pub prefix_tokens_saved: u64,
    pub cow_copies: u64,
    pub max_observed_active_sequences: usize,
    pub max_observed_queued_requests: usize,
    #[serde(default)]
    pub max_observed_prefilling_requests: usize,
    #[serde(default)]
    pub max_observed_pending_prefill_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub kv_reserved_bytes: usize,
    pub total_blocks: usize,
    /// Exact allocator high water, including allocations freed inside a step.
    pub peak_allocated_blocks: usize,
    /// Includes resident prefix-cache blocks. This is not process RSS.
    pub peak_occupied_kv_bytes: usize,
    pub final_allocated_blocks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayReport {
    pub clock: ReplayClock,
    pub elapsed_ms: f64,
    pub requests: Vec<RequestRecord>,
    pub summary: ReplaySummary,
    pub engine: EngineCounters,
    pub memory: MemoryStats,
}

#[derive(Default)]
struct TrackedSequence {
    tokens: Vec<u32>,
    deliveries: Vec<Delivery>,
    terminal: Option<FinishReason>,
    completion: Option<Completion>,
    finished_at_ms: Option<f64>,
}

struct TrackedRequest {
    id: String,
    scheduled_at: u64,
    submitted_at_ms: f64,
    origin_ms: f64,
    engine_id: Option<usize>,
    samples: usize,
    sequences: BTreeMap<usize, TrackedSequence>,
    terminal: Option<(RequestStatus, f64)>,
    rejection: Option<String>,
    cancel: Option<(u64, f64)>,
}

impl TrackedRequest {
    fn finish(self, clock: ReplayClock) -> Result<RequestRecord> {
        let (status, finished_at_ms) = self
            .terminal
            .with_context(|| format!("request {} never terminated", self.id))?;
        let mut sequences = Vec::new();
        for (sample_index, seq) in self.sequences.into_values().enumerate() {
            let c = seq.completion.context("sequence missing completion")?;
            ensure!(
                seq.terminal == Some(c.finish_reason),
                "terminal/completion mismatch"
            );
            ensure!(seq.tokens == c.tokens, "stream/completion token mismatch");
            let first = seq.deliveries.first();
            let finished_at_ms = seq.finished_at_ms.context("missing terminal timestamp")?;
            sequences.push(SequenceRecord {
                sample_index,
                tokens: seq.tokens,
                finish_reason: c.finish_reason,
                ttft_ms: first.map(|d| (d.at_ms - self.origin_ms).max(0.0)),
                engine_ttft_ms: first.map(|_| ms(c.time_to_first_token)),
                engine_queue_ms: ms(c.queue_time),
                end_to_end_ms: (finished_at_ms - self.origin_ms).max(0.0),
                finished_at_ms,
                deliveries: seq.deliveries,
            });
        }
        Ok(RequestRecord {
            id: self.id,
            scheduled_at: self.scheduled_at,
            submitted_at_ms: self.submitted_at_ms,
            dispatch_lag_ms: lag(clock, self.scheduled_at, self.submitted_at_ms),
            cancel_scheduled_at: self.cancel.map(|c| c.0),
            cancel_dispatched_at_ms: self.cancel.map(|c| c.1),
            cancel_dispatch_lag_ms: self.cancel.and_then(|(at, now)| lag(clock, at, now)),
            status,
            rejection: self.rejection,
            finished_at_ms,
            sequences,
        })
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
fn lag(clock: ReplayClock, at: u64, now: f64) -> Option<f64> {
    (clock == ReplayClock::Milliseconds).then(|| (now - at as f64).max(0.0))
}

fn drain(
    engine: &mut Engine<'_>,
    requests: &mut [TrackedRequest],
    engine_ids: &HashMap<usize, usize>,
    now_ms: f64,
) -> Result<()> {
    // Admission and decode can both emit in one step. Coalesce the tokens that
    // become observable in this drain into one batch per sequence.
    let mut batches: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for delta in engine.take_deltas() {
        let index = *engine_ids
            .get(&delta.request_id)
            .context("delta for unknown request")?;
        let seq = requests[index]
            .sequences
            .entry(delta.sequence_id)
            .or_default();
        ensure!(
            seq.terminal.is_none(),
            "delta after terminal for request {}",
            requests[index].id
        );
        let count = delta.tokens.len();
        seq.tokens.extend(delta.tokens);
        if count > 0 {
            *batches.entry((index, delta.sequence_id)).or_default() += count;
        }
        if let Some(reason) = delta.finish_reason {
            seq.terminal = Some(reason);
            seq.finished_at_ms = Some(now_ms);
        }
    }
    for ((index, sid), tokens) in batches {
        requests[index]
            .sequences
            .get_mut(&sid)
            .expect("delta created sequence")
            .deliveries
            .push(Delivery {
                at_ms: now_ms,
                tokens,
            });
    }
    for completion in engine.take_completed() {
        let index = *engine_ids
            .get(&completion.request_id)
            .context("completion for unknown request")?;
        let req = &mut requests[index];
        let seq = req
            .sequences
            .get_mut(&completion.sequence_id)
            .context("completion without streaming delta (enable stream_tokens)")?;
        ensure!(
            seq.completion.is_none(),
            "duplicate completion for request {}",
            req.id
        );
        ensure!(
            seq.terminal == Some(completion.finish_reason),
            "completion missing matching terminal delta"
        );
        ensure!(
            seq.tokens == completion.tokens,
            "stream/completion tokens differ for request {}",
            req.id
        );
        seq.completion = Some(completion);
        let completed = req
            .sequences
            .values()
            .filter(|s| s.completion.is_some())
            .count();
        ensure!(
            completed <= req.samples,
            "too many sequences for request {}",
            req.id
        );
        if completed == req.samples {
            req.terminal = Some((RequestStatus::Finished, now_ms));
        }
    }
    Ok(())
}

/// Replay on an empty streaming engine. Call `warm_up`/`reset` before this call;
/// setup, model loading, and warmup are excluded from measured elapsed time.
/// Limits are checked between steps, because an engine step is synchronous.
/// Elapsed time spans the entire trace, including idle gaps and late no-op
/// cancellations, through the final event and all outstanding completions.
pub fn run(
    engine: &mut Engine<'_>,
    workload: &Workload,
    limits: &ReplayLimits,
) -> Result<ReplayReport> {
    workload.validate()?;
    ensure!(
        engine.is_streaming(),
        "replay requires EngineConfig.stream_tokens=true"
    );
    ensure!(
        !engine.has_work() && engine.stats().steps == 0 && engine.stats().requests == 0,
        "replay requires an empty reset engine"
    );
    ensure!(
        limits.max_steps > 0 && !limits.timeout.is_zero(),
        "replay limits must be positive"
    );
    // These must not silently become events in this run after a prior cancellation.
    ensure!(
        engine.take_completed().is_empty() && engine.take_deltas().is_empty(),
        "reset the engine before replay: pending output exists"
    );
    let mut requests = Vec::<TrackedRequest>::new();
    let mut ids = HashMap::<String, usize>::new();
    let mut engine_ids = HashMap::<usize, usize>::new();
    let mut event_index = 0;
    let mut tick = 0_u64;
    let mut executed_steps = 0;
    let mut max_active = 0;
    let mut max_queued = 0;
    let mut max_prefilling = 0;
    let mut max_pending_prefill_tokens = 0;
    let start = Instant::now();
    while event_index < workload.events.len() || engine.has_work() {
        ensure!(
            start.elapsed() < limits.timeout,
            "replay timeout after {executed_steps} engine steps"
        );
        while let Some(event) = workload.events.get(event_index) {
            let elapsed = start.elapsed();
            let due = match workload.clock {
                ReplayClock::Steps => tick >= event.at(),
                ReplayClock::Milliseconds => elapsed >= Duration::from_millis(event.at()),
            };
            if !due {
                break;
            }
            let now_ms = ms(elapsed);
            match event {
                ReplayEvent::Submit {
                    at,
                    id,
                    tokens,
                    max_tokens,
                    samples,
                    seed,
                    temperature,
                    top_p,
                    top_k,
                } => {
                    let result = engine.submit_tokens_with(
                        tokens.clone(),
                        *max_tokens,
                        *samples,
                        RequestOptions {
                            seed: Some(*seed),
                            temperature: Some(*temperature),
                            top_p: Some(*top_p),
                            top_k: Some(*top_k),
                        },
                    );
                    let mut req = TrackedRequest {
                        id: id.clone(),
                        scheduled_at: *at,
                        submitted_at_ms: now_ms,
                        origin_ms: if workload.clock == ReplayClock::Milliseconds {
                            *at as f64
                        } else {
                            now_ms
                        },
                        engine_id: None,
                        samples: *samples,
                        sequences: BTreeMap::new(),
                        terminal: None,
                        rejection: None,
                        cancel: None,
                    };
                    match result {
                        Ok(engine_id) => {
                            req.engine_id = Some(engine_id);
                            engine_ids.insert(engine_id, requests.len());
                        }
                        Err(error) => {
                            req.terminal = Some((RequestStatus::Rejected, ms(start.elapsed())));
                            req.rejection = Some(error.to_string());
                        }
                    }
                    ids.insert(id.clone(), requests.len());
                    requests.push(req);
                }
                ReplayEvent::Cancel { at, id } => {
                    let index = ids[id];
                    let req = &mut requests[index];
                    req.cancel = Some((*at, now_ms));
                    if req.terminal.is_none() {
                        let stopped = engine.cancel_request(
                            req.engine_id
                                .context("accepted request missing engine id")?,
                        );
                        drain(engine, &mut requests, &engine_ids, ms(start.elapsed()))?;
                        // An admitted partial prefill can cancel before producing
                        // tokens, but still emits empty per-sample completions.
                        // Only never-admitted requests need a synthetic outcome.
                        if requests[index].sequences.is_empty() && stopped > 0 {
                            requests[index].terminal = Some((
                                RequestStatus::CancelledBeforeAdmission,
                                ms(start.elapsed()),
                            ));
                        }
                    }
                }
            }
            event_index += 1;
            let (active, queued) = engine.queue_depth();
            max_active = max_active.max(active);
            max_queued = max_queued.max(queued);
            max_prefilling = max_prefilling.max(engine.prefilling_requests());
            max_pending_prefill_tokens =
                max_pending_prefill_tokens.max(engine.pending_prefill_tokens());
            ensure!(
                start.elapsed() < limits.timeout,
                "replay timeout while dispatching events"
            );
        }
        if engine.has_work() {
            ensure!(
                executed_steps < limits.max_steps,
                "replay exceeded {} engine steps",
                limits.max_steps
            );
            engine.step()?;
            executed_steps += 1;
            tick = tick.checked_add(1).context("step clock overflow")?;
            drain(engine, &mut requests, &engine_ids, ms(start.elapsed()))?;
            let (active, queued) = engine.queue_depth();
            max_active = max_active.max(active);
            max_queued = max_queued.max(queued);
            max_prefilling = max_prefilling.max(engine.prefilling_requests());
            max_pending_prefill_tokens =
                max_pending_prefill_tokens.max(engine.pending_prefill_tokens());
        } else if let Some(next) = workload.events.get(event_index) {
            match workload.clock {
                // Idle gaps need no engine calls; this is a dispatch boundary,
                // not a count of fake decode operations.
                ReplayClock::Steps => tick = next.at(),
                ReplayClock::Milliseconds => {
                    let remaining =
                        Duration::from_millis(next.at()).saturating_sub(start.elapsed());
                    let budget = limits.timeout.saturating_sub(start.elapsed());
                    std::thread::sleep(remaining.min(budget).min(Duration::from_millis(50)));
                }
            }
        }
    }
    let elapsed_ms = ms(start.elapsed());
    ensure!(
        start.elapsed() < limits.timeout,
        "replay timeout during final engine step"
    );
    let requests = requests
        .into_iter()
        .map(|r| r.finish(workload.clock))
        .collect::<Result<Vec<_>>>()?;
    let summary = summarize(&requests, elapsed_ms);
    let stats = engine.stats();
    let prefix = engine.prefix_stats();
    let peak = engine.peak_allocated_blocks();
    let memory = MemoryStats {
        kv_reserved_bytes: engine.kv_cache_bytes(),
        total_blocks: engine.total_blocks(),
        peak_allocated_blocks: peak,
        peak_occupied_kv_bytes: engine.kv_cache_bytes() / engine.total_blocks() * peak,
        final_allocated_blocks: engine.total_blocks() - engine.available_blocks(),
    };
    Ok(ReplayReport {
        clock: workload.clock,
        elapsed_ms,
        requests,
        summary,
        memory,
        engine: EngineCounters {
            steps: stats.steps,
            admitted_requests: stats.requests,
            prompt_tokens: stats.prompt_tokens,
            prompt_tokens_prefilled: stats.prompt_tokens_prefilled,
            prompt_tokens_reused: stats.prompt_tokens_reused(),
            generated_tokens: stats.generated_tokens,
            prefill_ms: ms(stats.prefill_time),
            decode_ms: ms(stats.decode_time),
            preemptions: stats.preemptions,
            prefill_preemptions: stats.prefill_preemptions,
            prefill_chunks: stats.prefill_chunks,
            last_prefill_tokens: stats.last_prefill_tokens,
            recomputed_tokens: stats.recomputed_tokens,
            deferred_steps: stats.deferred_steps,
            prefix_hits: prefix.hits,
            prefix_lookups: prefix.hits + prefix.misses,
            prefix_tokens_saved: prefix.tokens_saved,
            cow_copies: engine.cow_copies(),
            max_observed_active_sequences: max_active,
            max_observed_queued_requests: max_queued,
            max_observed_prefilling_requests: max_prefilling,
            max_observed_pending_prefill_tokens: max_pending_prefill_tokens,
        },
    })
}

fn summarize(requests: &[RequestRecord], elapsed_ms: f64) -> ReplaySummary {
    let mut completed = 0;
    let mut cancelled = 0;
    let mut oom = 0;
    let mut successful_requests = 0;
    let mut cancelled_requests = 0;
    let mut rejected_requests = 0;
    let mut failed_requests = 0;
    let mut output_tokens = 0;
    let mut useful_tokens = 0;
    let mut ttft = Vec::new();
    let mut queue = Vec::new();
    let mut e2e = Vec::new();
    let mut token_gaps = Vec::new();
    let mut delivery_gaps = Vec::new();
    for req in requests {
        match req.status {
            RequestStatus::Rejected => rejected_requests += 1,
            RequestStatus::CancelledBeforeAdmission => cancelled_requests += 1,
            RequestStatus::Finished => {
                if req
                    .sequences
                    .iter()
                    .any(|s| s.finish_reason == FinishReason::OutOfMemory)
                {
                    failed_requests += 1;
                } else if req
                    .sequences
                    .iter()
                    .any(|s| s.finish_reason == FinishReason::Cancelled)
                {
                    cancelled_requests += 1;
                } else {
                    successful_requests += 1;
                }
            }
        }
        for seq in &req.sequences {
            output_tokens += seq.tokens.len();
            match seq.finish_reason {
                FinishReason::Eos | FinishReason::Length => {
                    completed += 1;
                    useful_tokens += seq.tokens.len();
                }
                FinishReason::Cancelled => cancelled += 1,
                FinishReason::OutOfMemory => oom += 1,
            }
            if let Some(first) = seq.ttft_ms {
                ttft.push(first);
            }
            queue.push(seq.engine_queue_ms);
            e2e.push(seq.end_to_end_ms);
            let mut last = None;
            for batch in &seq.deliveries {
                if let Some(prior) = last {
                    let gap = batch.at_ms - prior;
                    delivery_gaps.push(gap);
                    token_gaps.push(gap);
                }
                token_gaps.extend(std::iter::repeat_n(0.0, batch.tokens.saturating_sub(1)));
                last = Some(batch.at_ms);
            }
        }
    }
    let seconds = elapsed_ms / 1000.0;
    ReplaySummary {
        requests: requests.len(),
        successful_requests,
        rejected_requests,
        cancelled_requests,
        failed_requests,
        completed_sequences: completed,
        cancelled_sequences: cancelled,
        oom_sequences: oom,
        output_tokens,
        useful_output_tokens: useful_tokens,
        completed_requests_per_second: successful_requests as f64 / seconds.max(1e-9),
        useful_tokens_per_second: useful_tokens as f64 / seconds.max(1e-9),
        dispatch_lag_ms: Distribution::from_samples(
            requests.iter().filter_map(|r| r.dispatch_lag_ms).collect(),
        ),
        ttft_ms: Distribution::from_samples(ttft),
        engine_queue_ms: Distribution::from_samples(queue),
        end_to_end_ms: Distribution::from_samples(e2e),
        observed_inter_token_ms: Distribution::from_samples(token_gaps),
        inter_delivery_ms: Distribution::from_samples(delivery_gaps),
    }
}

/// Compare stable workload IDs/sample indices, full token vectors, and finish
/// reasons. Wall-clock cancellation and different tokens-per-step schedulers may
/// legitimately differ: callers must opt in, never silently exclude those rows.
pub fn compare_outputs(expected: &ReplayReport, actual: &ReplayReport) -> Result<()> {
    let left: BTreeMap<_, _> = expected.requests.iter().map(|r| (&r.id, r)).collect();
    let right: BTreeMap<_, _> = actual.requests.iter().map(|r| (&r.id, r)).collect();
    ensure!(
        left.len() == expected.requests.len() && right.len() == actual.requests.len(),
        "duplicate report request IDs"
    );
    ensure!(left.keys().eq(right.keys()), "request IDs differ");
    for (id, a) in left {
        let b = right[id];
        ensure!(
            a.status == b.status && a.rejection == b.rejection,
            "request {id}: outcome differs"
        );
        ensure!(
            a.sequences.len() == b.sequences.len(),
            "request {id}: sample count differs"
        );
        for (a, b) in a.sequences.iter().zip(&b.sequences) {
            ensure!(
                a.sample_index == b.sample_index,
                "request {id}: sample identity differs"
            );
            ensure!(
                a.finish_reason == b.finish_reason,
                "request {id} sample {}: finish reason differs",
                a.sample_index
            );
            if a.tokens != b.tokens {
                let first = a
                    .tokens
                    .iter()
                    .zip(&b.tokens)
                    .position(|(x, y)| x != y)
                    .unwrap_or(a.tokens.len().min(b.tokens.len()));
                bail!(
                    "request {id} sample {}: tokens differ at offset {first} (lengths {} vs {})",
                    a.sample_index,
                    a.tokens.len(),
                    b.tokens.len()
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn older_engine_reports_default_new_prefill_metrics() {
        let old = serde_json::json!({
            "steps": 7, "admitted_requests": 2, "prompt_tokens": 40,
            "prompt_tokens_prefilled": 32, "generated_tokens": 16,
            "prefill_ms": 4.0, "decode_ms": 2.0, "preemptions": 0,
            "recomputed_tokens": 0, "deferred_steps": 0,
            "prefix_hits": 2, "prefix_lookups": 4, "prefix_tokens_saved": 8,
            "cow_copies": 0, "max_observed_active_sequences": 2,
            "max_observed_queued_requests": 1
        });
        let counters: EngineCounters = serde_json::from_value(old).unwrap();
        assert_eq!(counters.steps, 7);
        assert_eq!(counters.prompt_tokens_prefilled, 32);
        assert_eq!(counters.prompt_tokens_reused, 0);
        assert_eq!(counters.prefill_chunks, 0);
        assert_eq!(counters.prefill_preemptions, 0);
        assert_eq!(counters.last_prefill_tokens, 0);
        assert_eq!(counters.max_observed_prefilling_requests, 0);
        assert_eq!(counters.max_observed_pending_prefill_tokens, 0);
    }

    #[test]
    fn nearest_rank_percentiles_preserve_empty_and_single_populations() {
        let empty = Distribution::from_samples(vec![]);
        assert_eq!(empty.count, 0);
        assert_eq!(empty.p50, None);
        let one = Distribution::from_samples(vec![7.0]);
        assert_eq!(
            (one.p50, one.p95, one.p99),
            (Some(7.0), Some(7.0), Some(7.0))
        );
        let many = Distribution::from_samples((1..=100).rev().map(f64::from).collect());
        assert_eq!(
            (many.p50, many.p95, many.p99),
            (Some(50.0), Some(95.0), Some(99.0))
        );
    }

    #[test]
    fn malformed_workloads_are_refused() {
        let base = serde_json::json!({"version":1,"clock":"steps","events":[
            {"type":"submit","at":0,"id":"a","tokens":[1],"max_tokens":3,"seed":42}
        ]});
        let good: Workload = serde_json::from_value(base.clone()).unwrap();
        good.validate().unwrap();
        for (field, value) in [
            ("samples", serde_json::json!(0)),
            ("top_p", serde_json::json!(1.1)),
            ("temperature", serde_json::json!(-1)),
            ("max_tokens", serde_json::json!(0)),
        ] {
            let mut value_bad = base.clone();
            value_bad["events"][0][field] = value;
            assert!(serde_json::from_value::<Workload>(value_bad)
                .unwrap()
                .validate()
                .is_err());
        }
        let mut missing_seed = base.clone();
        missing_seed["events"][0]
            .as_object_mut()
            .unwrap()
            .remove("seed");
        assert!(serde_json::from_value::<Workload>(missing_seed).is_err());
        let mut typo = base;
        typo["events"][0]["max_token"] = 4.into();
        assert!(serde_json::from_value::<Workload>(typo).is_err());
        let mut duplicate = good.clone();
        duplicate.events.push(good.events[0].clone());
        assert!(duplicate.validate().is_err());
        let mut early_cancel = good;
        early_cancel.events.insert(
            0,
            ReplayEvent::Cancel {
                at: 0,
                id: "a".into(),
            },
        );
        assert!(early_cancel.validate().is_err());
    }
}
