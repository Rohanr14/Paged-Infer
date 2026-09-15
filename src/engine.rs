//! The serving loop: iteration-level scheduling over a paged KV cache.
//!
//! One `step()` advances existing decoders, reclaims finished allocations, then
//! performs a bounded amount of prefill. A retained FIFO prompt cursor lets the
//! caller deliver tokens and handle cancellation between prompt slices. Requests
//! whose prefill completes begin ordinary decode on the following step.
//!
//! Deliberately independent of tokenization: the scheduler works on token ids,
//! and text is a thin convenience layer on top. That keeps the whole engine
//! testable against a synthetic checkpoint without dragging in a real tokenizer.
//!
//! # Memory discipline
//!
//! Every KV write a step performs is either *mandatory* — the token a sequence
//! already holds must land somewhere, or the sequence cannot advance at all —
//! or *optional*, which is speculative drafting. The planner settles every
//! mandatory write for every sequence before it spends a single block on an
//! optional one, so a guess can never take the block a sibling needs.
//!
//! A sequence whose mandatory write cannot be satisfied does not write. It
//! **defers**: it keeps its blocks, sits the step out, and is retried next
//! step, when whatever finished this step has freed memory. Writing anyway is
//! never an option — the block it would write is shared with a sibling or the
//! prefix cache, and writing through it silently hands every other holder this
//! sequence's KV state. When *no* sequence can advance the step would make no
//! progress at all, so the planner **preempts**: the most recently created
//! blocked sequence gives up its blocks and goes back to the head of the queue
//! to be recomputed from its own tokens when memory allows. Only a sequence
//! that could never be re-admitted — one whose tokens no longer fit in an empty
//! pool — is terminated, with [`FinishReason::OutOfMemory`].
//!
//! Admission plays by the same rule: a prompt is admitted only if, after its
//! blocks are mapped, enough remain free for every live sequence's next
//! mandatory write and for the prompt's own first decode step. Without that a
//! prompt could prefill and then strand a running sequence — or itself — one
//! block short at the very next step.

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::memory::block_table::BlockTable;
use crate::memory::kv_cache_manager::KvCacheManager;
use crate::memory::layout::KvLayout;
use crate::memory::prefix_cache::PrefixCacheStats;
use crate::model::{BatchScratch, LlamaConfig, LlamaWeights};
use crate::sampling::Sampler;
use crate::speculative::{verify_greedy, Drafter, PromptLookupDrafter, SpecStats};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    pub total_blocks: usize,
    pub block_size: usize,
    /// `0.0` is greedy. Multi-sample requests that do not ask for a
    /// temperature themselves override this upward, since identical greedy
    /// branches would make forking pointless; a request that explicitly asks
    /// for `0.0` gets it.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub eos_token: u32,
    /// Further token ids that end generation, for checkpoints with more than
    /// one end-of-turn token.
    pub extra_eos_tokens: Vec<u32>,
    /// Prepended to *text* prompts ([`Engine::submit`]) when the tokenizer did
    /// not already add it. Token-id prompts ([`Engine::submit_tokens`]) are
    /// complete model inputs and are never modified: the caller owns their
    /// special tokens.
    pub bos_token: Option<u32>,
    /// Longest sequence — prompt plus generated tokens — the model supports.
    /// Prompts beyond it are rejected at submission; generation stops with
    /// [`FinishReason::Length`] when it is reached. `None` leaves only the KV
    /// pool as a limit.
    pub max_context: Option<usize>,
    pub enable_prefix_cache: bool,
    /// Most sequences decoded in one batched pass. Larger batches amortize the
    /// weight traffic over more sequences; the cap bounds scratch memory, which
    /// grows as `max_batch_size * vocab_size`.
    pub max_batch_size: usize,
    /// Prompt positions pushed through the model together during prefill. Same
    /// trade as `max_batch_size`, along the position axis instead of the
    /// sequence axis.
    pub prefill_chunk_size: usize,
    /// Maximum prompt positions computed across all requests in one step.
    /// Existing decoders run first and do not spend this allowance. This is a
    /// prefill work bound, not a bound on total decode work or elapsed time.
    /// Separate from `prefill_chunk_size`, which controls matrix batching.
    pub max_prefill_tokens_per_step: usize,
    /// Experimental shared-prefix decode attention; unrelated batches fall back.
    /// Disabled by default pending the end-to-end performance gate.
    pub shared_prefix_attention: bool,
    /// Draft tokens to propose per step. `0` disables speculative decoding.
    ///
    /// Only greedy sequences speculate: acceptance is defined as "the model
    /// would have chosen this token", which is exactly greedy. A sampled
    /// sequence would need the rejection-sampling correction to stay
    /// distributionally faithful, so it takes the ordinary path instead of a
    /// subtly wrong shortcut.
    pub draft_tokens: usize,
    /// Record every token the moment it is produced, for
    /// [`Engine::take_deltas`].
    ///
    /// Off by default because it is only useful to a caller that drains it
    /// every step; a batch caller that never looked would accumulate the whole
    /// run's output twice over.
    pub stream_tokens: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            total_blocks: 512,
            block_size: 16,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0x5EED,
            eos_token: 2,
            extra_eos_tokens: Vec::new(),
            bos_token: Some(1),
            max_context: None,
            enable_prefix_cache: true,
            max_batch_size: 32,
            prefill_chunk_size: 32,
            max_prefill_tokens_per_step: 32,
            shared_prefix_attention: false,
            draft_tokens: 0,
            stream_tokens: false,
        }
    }
}

/// Per-request sampling overrides. `None` means "the engine default".
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RequestOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    /// Makes a request replayable on its own: the streams of its samples are
    /// derived from this seed and the sample index alone, never from what else
    /// the engine happened to be running.
    pub seed: Option<u64>,
}

/// Why a request was refused at submission. Every variant is the caller's
/// input being wrong, never an engine failure — a server maps these to a
/// client error, not a 500.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitError {
    EmptyPrompt,
    ZeroMaxTokens,
    /// A token id at or beyond the vocabulary. Never mapped or wrapped: the
    /// caller meant something that does not exist.
    InvalidToken {
        index: usize,
        token: u32,
        vocab_size: usize,
    },
    /// Longer than the model's context window.
    ExceedsContext {
        prompt_tokens: usize,
        max_context: usize,
    },
    /// The prompt (plus the first decode step of every sample) needs more
    /// blocks than the whole pool holds. It could never be admitted, so it is
    /// refused now rather than queued to stall everything behind it.
    DoesNotFit {
        prompt_tokens: usize,
        num_samples: usize,
        blocks_needed: usize,
        total_blocks: usize,
        block_size: usize,
    },
}

impl fmt::Display for SubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubmitError::EmptyPrompt => write!(f, "cannot generate from an empty prompt"),
            SubmitError::ZeroMaxTokens => write!(f, "max_tokens must be at least 1"),
            SubmitError::InvalidToken {
                index,
                token,
                vocab_size,
            } => write!(
                f,
                "token id {token} at position {index} is outside the vocabulary of {vocab_size}"
            ),
            SubmitError::ExceedsContext {
                prompt_tokens,
                max_context,
            } => write!(
                f,
                "a prompt of {prompt_tokens} tokens exceeds the model's context of {max_context}"
            ),
            SubmitError::DoesNotFit {
                prompt_tokens,
                num_samples,
                blocks_needed,
                total_blocks,
                block_size,
            } => write!(
                f,
                "a prompt of {prompt_tokens} tokens with {num_samples} sample(s) needs \
                 {blocks_needed} KV blocks and does not fit in {total_blocks} blocks of \
                 {block_size} tokens"
            ),
        }
    }
}

impl std::error::Error for SubmitError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model emitted an end-of-sequence token.
    Eos,
    /// The request's token budget, or the model's context window, ran out.
    Length,
    /// The KV cache could not hold another token, and the sequence could not
    /// be resumed later either: even an empty pool would not hold what it had
    /// already produced. Whatever was generated up to that point is returned.
    OutOfMemory,
    /// The caller gave up on the request — a disconnected streaming client,
    /// typically. Whatever was generated up to that point is still returned.
    Cancelled,
}

/// Tokens one sequence produced during a single step.
///
/// Usually one token. A speculative step that had drafts accepted emits the
/// whole accepted run at once, which is exactly what a streaming client should
/// see — the tokens really are all available at that instant.
///
/// Every sequence emits **exactly one** delta with `finish_reason` set, however
/// it ends: end-of-sequence, budget, memory, or cancellation. A terminal delta
/// may carry no tokens.
#[derive(Debug, Clone)]
pub struct TokenDelta {
    pub request_id: usize,
    pub sequence_id: usize,
    pub tokens: Vec<u32>,
    /// Set on the last delta of a sequence, so a streaming consumer can close
    /// the stream without waiting for the completion to be reclaimed.
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub request_id: usize,
    pub sequence_id: usize,
    pub prompt_tokens: usize,
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    /// Submission through to the first sampled token: what the client waited,
    /// queueing included. Admission-to-first-token alone flattered a saturated
    /// engine, where most of the wait is the queue.
    /// If `tokens` is empty (for example, cancellation during prefill), no first
    /// token exists: this stores the admission wait and is not a TTFT sample.
    pub time_to_first_token: Duration,
    /// Submission through to admission — the part of `time_to_first_token`
    /// spent waiting for initial admission, including memory and FIFO delays.
    pub queue_time: Duration,
}

#[derive(Debug, Default, Clone)]
pub struct RunStats {
    pub requests: usize,
    pub prompt_tokens: usize,
    /// Actual fresh-request prompt positions processed, including retries of
    /// discarded partial prefills. May exceed `prompt_tokens`; unfinished or
    /// cancelled suffixes have not been processed. Cache reuse is explicit.
    pub prompt_tokens_prefilled: usize,
    /// Positions actually skipped via prefix reuse on each fresh request's
    /// initial admission. Uncomputed/cancelled suffixes are never cache hits.
    pub prompt_tokens_reused: usize,
    /// Unfinished prompt mappings discarded to make mandatory decode writable.
    pub prefill_preemptions: usize,
    /// Scheduler slices processed, including resumption and retries.
    pub prefill_chunks: usize,
    /// Actual prefill positions computed in the most recent engine step.
    pub last_prefill_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub steps: usize,
    /// Sequences that gave their blocks up under memory pressure and were
    /// queued to be recomputed.
    pub preemptions: usize,
    /// Inputs processed for resumed decoding sequences, plus repeated positions
    /// after partial-prefill eviction. The latter overlap with
    /// `prompt_tokens_prefilled`; these counters must not be summed as disjoint
    /// populations.
    pub recomputed_tokens: usize,
    /// Sequence-steps that sat out because their next KV write had no block.
    pub deferred_steps: usize,
}

impl RunStats {
    pub fn prompt_tokens_reused(&self) -> usize {
        self.prompt_tokens_reused
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        self.generated_tokens as f64 / self.decode_time.as_secs_f64().max(1e-9)
    }
}

/// State a preempted sequence carries back to the queue, so that when it is
/// re-admitted it continues rather than restarts: same sequence id (a
/// streaming client's choice index depends on it), same sampler state, same
/// output so far.
struct Preempted {
    sequence_id: usize,
    prompt_len: usize,
    generated: Vec<u32>,
    sampler: Sampler,
    drafter: Option<Box<dyn Drafter>>,
    admitted_at: Instant,
    first_token_at: Option<Instant>,
}

struct Request {
    id: usize,
    tokens: Vec<u32>,
    max_tokens: usize,
    num_samples: usize,
    options: RequestOptions,
    submitted_at: Instant,
    /// Set when this entry re-admits a preempted sequence. Its `tokens` are
    /// then the sequence's prompt *and* everything it generated, and the
    /// prefill recomputes the KV the sequence gave up.
    resume: Option<Preempted>,
    /// Assigned once on first admission, before prefill can yield. Retained
    /// across partial-prefill eviction so default RNG streams and sample IDs
    /// do not change when memory pressure changes the schedule.
    sequence_ids: Vec<usize>,
    admitted_at: Option<Instant>,
    /// Furthest prompt position previously computed (or borrowed from cache).
    /// Reprocessing earlier positions after eviction is counted as recompute.
    prefill_high_water: usize,
}

/// One prompt owns the FIFO prefill lane until completion or cancellation.
/// All its blocks are mapped up front, but new hashes are published only once
/// every layer's KV for the entire prompt is complete.
struct Prefilling {
    request: Request,
    block_table: BlockTable,
    cursor: usize,
}

struct Sequence {
    id: usize,
    request_id: usize,
    prompt_len: usize,
    token_ids: Vec<u32>,
    generated: Vec<u32>,
    max_tokens: usize,
    block_table: BlockTable,
    finished: Option<FinishReason>,
    sampler: Sampler,
    submitted_at: Instant,
    admitted_at: Instant,
    first_token_at: Option<Instant>,
    /// Present only when this sequence speculates.
    drafter: Option<Box<dyn Drafter>>,
    /// The last step could not secure the block this sequence's next write
    /// needs. It sat the step out and still holds its blocks.
    deferred: bool,
    /// Gave its blocks up this step; `reclaim` moves it back to the queue.
    preempted: bool,
}

/// One sequence's contribution to a decode step: the token it already holds,
/// plus whatever its drafter guessed comes next.
struct SpecGroup {
    seq_idx: usize,
    /// `[held_token, draft_0, ..]` — what the model is asked to process.
    tokens: Vec<u32>,
    positions: Vec<usize>,
    /// `tokens[1..]`, kept separately because verification only concerns these.
    drafts: Vec<u32>,
}

/// Why a sequence stopped, if it did, after `token` was appended.
fn stop_reason(
    eos: u32,
    extra_eos: &[u32],
    max_context: Option<usize>,
    token: u32,
    generated_len: usize,
    max_tokens: usize,
    seq_len: usize,
) -> Option<FinishReason> {
    if token == eos || extra_eos.contains(&token) {
        Some(FinishReason::Eos)
    } else if generated_len >= max_tokens {
        Some(FinishReason::Length)
    } else if max_context.is_some_and(|limit| seq_len > limit) {
        // The token just appended would have to be written at position
        // `seq_len - 1`, which the model has no rotary table for.
        Some(FinishReason::Length)
    } else {
        None
    }
}

pub struct Engine<'a> {
    weights: LlamaWeights<'a>,
    config: LlamaConfig,
    engine: EngineConfig,
    kv: KvCacheManager,
    kv_cache: Vec<f32>,
    layout: KvLayout,
    batch_scratch: BatchScratch,
    tokenizer: Option<Tokenizer>,
    waiting: VecDeque<Request>,
    prefilling: Option<Prefilling>,
    active: Vec<Sequence>,
    completed: Vec<Completion>,
    deltas: Vec<TokenDelta>,
    next_request_id: usize,
    next_sequence_id: usize,
    tick: u64,
    stats: RunStats,
    spec: SpecStats,
}

impl<'a> Engine<'a> {
    pub fn new(weights: LlamaWeights<'a>, config: LlamaConfig, engine: EngineConfig) -> Self {
        assert!(engine.block_size > 0, "block_size must be at least 1");
        assert!(engine.total_blocks > 0, "total_blocks must be at least 1");
        assert!(
            engine.max_prefill_tokens_per_step > 0,
            "prefill token budget must be at least 1"
        );
        let layout = config.kv_layout(engine.total_blocks, engine.block_size);
        let kv_cache = vec![0.0; layout.total_floats()];
        let kv = KvCacheManager::new(engine.total_blocks, engine.block_size)
            .with_prefix_cache(engine.enable_prefix_cache);
        let batch_capacity = engine.max_batch_size.max(engine.prefill_chunk_size).max(1);
        let mut batch_scratch = BatchScratch::new(&config, batch_capacity);
        batch_scratch.set_shared_prefix_attention(engine.shared_prefix_attention);

        Self {
            weights,
            config,
            engine,
            kv,
            kv_cache,
            layout,
            batch_scratch,
            tokenizer: None,
            waiting: VecDeque::new(),
            prefilling: None,
            active: Vec::new(),
            completed: Vec::new(),
            deltas: Vec::new(),
            next_request_id: 0,
            next_sequence_id: 0,
            tick: 0,
            stats: RunStats::default(),
            spec: SpecStats::default(),
        }
    }

    pub fn with_tokenizer(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    pub fn kv_cache_bytes(&self) -> usize {
        self.kv_cache.len() * std::mem::size_of::<f32>()
    }

    pub fn stats(&self) -> &RunStats {
        &self.stats
    }

    pub fn prefix_stats(&self) -> PrefixCacheStats {
        self.kv.prefix_stats()
    }

    pub fn cow_copies(&self) -> u64 {
        self.kv.cow_copies()
    }

    pub fn total_blocks(&self) -> usize {
        self.kv.total_blocks()
    }

    /// Blocks not currently mapped by a sequence or held by the prefix cache.
    pub fn available_blocks(&self) -> usize {
        self.kv.available_blocks()
    }

    /// Peak simultaneously occupied physical KV blocks, including transient
    /// allocations within a step and blocks retained by the prefix cache.
    /// Reset and warmup clear this counter. This does not measure process RSS
    /// or the fixed backing allocation reported by [`Self::kv_cache_bytes`].
    pub fn peak_allocated_blocks(&self) -> usize {
        self.kv.peak_allocated_blocks()
    }

    /// Whether generated tokens and terminal events are available as deltas.
    pub fn is_streaming(&self) -> bool {
        self.engine.stream_tokens
    }

    /// Sequences decoding right now, and requests still queued behind them.
    pub fn queue_depth(&self) -> (usize, usize) {
        (
            self.active.len(),
            self.waiting.len() + self.prefilling_requests(),
        )
    }

    pub fn prefilling_requests(&self) -> usize {
        usize::from(self.prefilling.is_some())
    }

    pub fn pending_prefill_tokens(&self) -> usize {
        self.prefilling
            .as_ref()
            .map_or(0, |p| p.request.tokens.len() - p.cursor)
    }

    /// Sequences that sat the last step out waiting for a KV block.
    pub fn deferred_sequences(&self) -> usize {
        self.active.iter().filter(|s| s.deferred).count()
    }

    /// The model's context window, or the pool's capacity if the model does
    /// not declare one. No prompt longer than this can be submitted.
    pub fn max_prompt_tokens(&self) -> usize {
        let pool = self.engine.total_blocks * self.engine.block_size;
        self.engine.max_context.map_or(pool, |c| c.min(pool))
    }

    /// Stop generating for a request, freeing its blocks at the next step.
    ///
    /// A streaming client that hangs up would otherwise keep paying for tokens
    /// nobody will read, and — worse on a small pool — keep holding KV blocks
    /// that a live request needs. Returns the number of sequences stopped.
    ///
    /// Already-generated tokens are kept and the sequence completes normally
    /// with [`FinishReason::Cancelled`], so the caller still gets a well-formed
    /// completion rather than a dangling request, and a streaming caller gets
    /// the terminal delta that closes its stream. A request that was never
    /// admitted simply disappears: it produced nothing to complete.
    pub fn cancel_request(&mut self, request_id: usize) -> usize {
        let mut stopped = 0;
        let mut kept = VecDeque::with_capacity(self.waiting.len());
        for req in std::mem::take(&mut self.waiting) {
            if req.id != request_id {
                kept.push_back(req);
            } else {
                stopped += self.cancel_unfinished_prefill(req);
            }
        }
        self.waiting = kept;
        if self
            .prefilling
            .as_ref()
            .is_some_and(|p| p.request.id == request_id)
        {
            let prefill = self.prefilling.take().expect("matching prefill");
            self.kv.release_sequence(prefill.request.sequence_ids[0]);
            stopped += self.cancel_unfinished_prefill(prefill.request);
        }
        for idx in 0..self.active.len() {
            let seq = &self.active[idx];
            if seq.request_id == request_id && seq.finished.is_none() && !seq.preempted {
                self.terminate(idx, FinishReason::Cancelled);
                stopped += 1;
            }
        }
        self.reclaim();
        stopped
    }

    /// Never-admitted requests retain the original request-level cancellation
    /// contract. Once admitted, every reserved sample gets one terminal event,
    /// even if prefill has not yet produced its first token.
    fn cancel_unfinished_prefill(&mut self, req: Request) -> usize {
        if let Some(p) = req.resume {
            self.record_delta(
                req.id,
                p.sequence_id,
                Vec::new(),
                Some(FinishReason::Cancelled),
            );
            self.completed.push(Completion {
                request_id: req.id,
                sequence_id: p.sequence_id,
                prompt_tokens: p.prompt_len,
                tokens: p.generated,
                finish_reason: FinishReason::Cancelled,
                time_to_first_token: p
                    .first_token_at
                    .unwrap_or(p.admitted_at)
                    .duration_since(req.submitted_at),
                queue_time: p.admitted_at.duration_since(req.submitted_at),
            });
            return 1;
        }
        let Some(admitted_at) = req.admitted_at else {
            return 1;
        };
        for &sid in &req.sequence_ids {
            self.record_delta(req.id, sid, Vec::new(), Some(FinishReason::Cancelled));
            self.completed.push(Completion {
                request_id: req.id,
                sequence_id: sid,
                prompt_tokens: req.tokens.len(),
                tokens: Vec::new(),
                finish_reason: FinishReason::Cancelled,
                // No sampled token exists. Consumers must check tokens before
                // reporting TTFT (the replay driver does so).
                time_to_first_token: admitted_at.duration_since(req.submitted_at),
                queue_time: admitted_at.duration_since(req.submitted_at),
            });
        }
        req.sequence_ids.len()
    }

    pub fn spec_stats(&self) -> SpecStats {
        self.spec
    }

    pub fn shared_attention_stats(&self) -> crate::model::SharedAttentionStats {
        self.batch_scratch.shared_attention_stats()
    }

    /// Change the depth used by active drafters. A fresh greedy sequence gets
    /// a drafter when its prefill completes if this depth is nonzero; changing
    /// it later does not add a drafter to an existing non-speculating sequence.
    /// Resumed sequences retain their original drafter.
    pub fn set_draft_tokens(&mut self, draft_tokens: usize) {
        self.engine.draft_tokens = draft_tokens;
    }

    /// Drop all queues, counters and cached KV, keeping the loaded weights.
    ///
    /// Lets a benchmark measure several configurations without paying to
    /// reload a multi-gigabyte checkpoint between them.
    pub fn reset(&mut self) {
        self.waiting.clear();
        self.prefilling = None;
        self.active.clear();
        self.completed.clear();
        self.deltas.clear();
        self.kv.clear();
        self.stats = RunStats::default();
        self.spec = SpecStats::default();
        self.batch_scratch.reset_shared_attention_stats();
        self.tick = 0;
    }

    /// Run one throwaway forward pass so the first real request does not absorb
    /// costs that belong to startup.
    ///
    /// Three one-time costs land on whichever request happens to be first: the
    /// rayon pool spinning up its worker threads, first-touch of the scratch
    /// arenas (which are `max_batch_size * vocab_size` floats and so are not
    /// small), and, for a memory-mapped checkpoint, a page fault per 4 KiB of
    /// weights. None of that is the first request's fault, and a server that
    /// reports time-to-first-token should not charge it for them.
    ///
    /// This runs a full prefill over a synthetic prompt — every layer, every
    /// projection — then returns the cache and all counters to their initial
    /// state, so a warmed engine is indistinguishable from a fresh one except
    /// for being warm. Must be called before any request is submitted.
    pub fn warm_up(&mut self) {
        assert!(
            !self.has_work() && self.stats.requests == 0,
            "warm_up must run before any request is submitted"
        );

        // Long enough to fill one prefill chunk, so the batched path is
        // exercised too, but never larger than the pool can hold.
        let capacity = self.engine.total_blocks * self.engine.block_size;
        let len = self.engine.prefill_chunk_size.clamp(2, capacity.max(2));
        let vocab = self.config.vocab_size as u32;
        let tokens: Vec<u32> = (0..len as u32).map(|i| i % vocab).collect();

        const WARMUP_SEQ: usize = usize::MAX;
        if let Some(admission) = self.kv.admit(WARMUP_SEQ, &tokens, 0) {
            self.weights.prefill_batched(
                &tokens,
                0,
                &self.config,
                &admission.block_table,
                &mut self.kv_cache,
                self.engine.block_size,
                self.engine.prefill_chunk_size,
                &mut self.batch_scratch,
            );
            self.kv.release_sequence(WARMUP_SEQ);
        }

        // The synthetic prompt must leave no trace: its blocks were never
        // published to the prefix cache, but `clear` also resets the hit/miss
        // counters so the reported hit rate is the workload's, not ours.
        self.kv.clear();
        self.stats = RunStats::default();
        self.spec = SpecStats::default();
        self.deltas.clear();
        self.tick = 0;
    }

    // ── submission ───────────────────────────────────────────────────────────

    /// Queue a prompt that is already tokenized. `num_samples` continuations are
    /// drawn from it; the prompt is prefilled once regardless.
    ///
    /// The ids are the complete model input: no BOS is inserted and nothing is
    /// remapped. Every id must be inside the vocabulary and the prompt must be
    /// one the pool can ever hold, or the request is refused here rather than
    /// queued — a queued request that can never be admitted would stall every
    /// request behind it.
    pub fn submit_tokens(
        &mut self,
        tokens: Vec<u32>,
        max_tokens: usize,
        num_samples: usize,
    ) -> Result<usize, SubmitError> {
        self.submit_tokens_with(tokens, max_tokens, num_samples, RequestOptions::default())
    }

    /// [`Engine::submit_tokens`] with per-request sampling options. A server
    /// needs this: temperature is a property of the request, not of the engine.
    pub fn submit_tokens_with(
        &mut self,
        tokens: Vec<u32>,
        max_tokens: usize,
        num_samples: usize,
        options: RequestOptions,
    ) -> Result<usize, SubmitError> {
        let num_samples = num_samples.max(1);
        self.validate(&tokens, max_tokens, num_samples)?;
        let id = self.next_request_id;
        self.next_request_id += 1;
        self.waiting.push_back(Request {
            id,
            tokens,
            max_tokens,
            num_samples,
            options,
            submitted_at: Instant::now(),
            resume: None,
            sequence_ids: Vec::new(),
            admitted_at: None,
            prefill_high_water: 0,
        });
        Ok(id)
    }

    fn validate(
        &self,
        tokens: &[u32],
        max_tokens: usize,
        num_samples: usize,
    ) -> Result<(), SubmitError> {
        if tokens.is_empty() {
            return Err(SubmitError::EmptyPrompt);
        }
        if max_tokens == 0 {
            return Err(SubmitError::ZeroMaxTokens);
        }
        let vocab_size = self.config.vocab_size;
        if let Some((index, &token)) = tokens
            .iter()
            .enumerate()
            .find(|(_, t)| **t as usize >= vocab_size)
        {
            return Err(SubmitError::InvalidToken {
                index,
                token,
                vocab_size,
            });
        }
        if let Some(max_context) = self.engine.max_context {
            if tokens.len() > max_context {
                return Err(SubmitError::ExceedsContext {
                    prompt_tokens: tokens.len(),
                    max_context,
                });
            }
        }
        let blocks_needed = self.first_step_blocks(tokens.len(), num_samples, max_tokens);
        if blocks_needed > self.engine.total_blocks {
            return Err(SubmitError::DoesNotFit {
                prompt_tokens: tokens.len(),
                num_samples,
                blocks_needed,
                total_blocks: self.engine.total_blocks,
                block_size: self.engine.block_size,
            });
        }
        Ok(())
    }

    /// Queue a text prompt. Requires a tokenizer.
    ///
    /// Special tokens are the tokenizer's business: it is asked to add them,
    /// and the engine's `bos_token` is prepended only if the tokenizer did not
    /// already put it there. Exactly one BOS reaches the model either way.
    pub fn submit(&mut self, prompt: &str, max_tokens: usize, num_samples: usize) -> Result<usize> {
        self.submit_with(prompt, max_tokens, num_samples, RequestOptions::default())
    }

    /// [`Engine::submit`] with per-request sampling options.
    pub fn submit_with(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        num_samples: usize,
        options: RequestOptions,
    ) -> Result<usize> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("engine has no tokenizer; use submit_tokens"))?;
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let mut ids = encoding.get_ids().to_vec();
        if let Some(bos) = self.engine.bos_token {
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        Ok(self.submit_tokens_with(ids, max_tokens, num_samples, options)?)
    }

    pub fn decode_text(&self, tokens: &[u32]) -> Option<String> {
        self.tokenizer.as_ref()?.decode(tokens, true).ok()
    }

    /// True while anything is queued or running.
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || self.prefilling.is_some() || !self.active.is_empty()
    }

    /// Take the completions finished since the last call.
    ///
    /// This is what lets a server interleave: submit whenever a client
    /// connects, `step()` continuously, and hand back each completion as it
    /// lands, rather than waiting for the whole batch to drain.
    pub fn take_completed(&mut self) -> Vec<Completion> {
        std::mem::take(&mut self.completed)
    }

    /// Take the tokens produced since the last call, per sequence.
    ///
    /// Empty unless [`EngineConfig::stream_tokens`] is set. This is what makes
    /// streaming responses possible without changing the scheduler: the loop
    /// still advances every sequence together, and a client that wants tokens
    /// as they land reads them here instead of waiting for the completion.
    pub fn take_deltas(&mut self) -> Vec<TokenDelta> {
        std::mem::take(&mut self.deltas)
    }

    fn record_delta(
        &mut self,
        request_id: usize,
        sequence_id: usize,
        tokens: Vec<u32>,
        finish_reason: Option<FinishReason>,
    ) {
        if !self.engine.stream_tokens || (tokens.is_empty() && finish_reason.is_none()) {
            return;
        }
        self.deltas.push(TokenDelta {
            request_id,
            sequence_id,
            tokens,
            finish_reason,
        });
    }

    /// End a live sequence for a reason other than its own output — memory or
    /// cancellation — and tell any streaming consumer so. This and `commit` are
    /// the only two places a sequence's `finished` is set, which is what keeps
    /// "exactly one terminal delta per sequence" true.
    fn terminate(&mut self, idx: usize, reason: FinishReason) {
        let seq = &mut self.active[idx];
        debug_assert!(
            seq.finished.is_none(),
            "sequence {} terminated twice",
            seq.id
        );
        if seq.finished.is_some() {
            return;
        }
        seq.finished = Some(reason);
        let (rid, sid) = (seq.request_id, seq.id);
        self.record_delta(rid, sid, Vec::new(), Some(reason));
    }

    /// Drain both queues, returning every completion in finish order.
    pub fn run(&mut self) -> Result<Vec<Completion>> {
        while self.has_work() {
            let waiting_before = self.waiting.len();
            let was_idle = self.active.is_empty() && self.prefilling.is_none();
            self.step()?;
            if was_idle
                && self.active.is_empty()
                && self.prefilling.is_none()
                && self.waiting.len() == waiting_before
            {
                // Nothing was running, nothing finished, and the head of the
                // queue still was not admitted. Submission refuses anything the
                // empty pool cannot hold, so this is an invariant failure, not a
                // condition the caller can hit — but a loop that spins forever
                // is the worst possible way to report one.
                anyhow::bail!(
                    "a queued prompt cannot be admitted into {} blocks of {} tokens \
                     even with the pool otherwise empty",
                    self.engine.total_blocks,
                    self.engine.block_size
                );
            }
        }
        Ok(std::mem::take(&mut self.completed))
    }

    /// Decode existing sequences, reclaim, then spend the bounded prefill
    /// allowance. Token delivery and cancellation can run between these calls.
    pub fn step(&mut self) -> Result<()> {
        self.tick += 1;
        self.stats.last_prefill_tokens = 0;
        self.decode();
        self.reclaim();
        self.advance_prefill();
        self.reclaim();
        self.stats.steps += 1;
        Ok(())
    }

    // ── memory arithmetic ────────────────────────────────────────────────────

    /// Blocks a prompt's first decode step will have to allocate beyond the
    /// prompt's own. The first sampled token comes off the prefill logits and
    /// costs nothing; the step after it writes that token's KV at position
    /// `prompt_len` for each of `num_samples` sequences — a fresh block each if
    /// the prompt ends on a block boundary, otherwise a private copy of the
    /// shared partial block for every sample but one. Zero when no decode step
    /// will follow: a budget of one token, or a prompt already at the context
    /// limit.
    fn first_step_growth(&self, prompt_len: usize, num_samples: usize, budget: usize) -> usize {
        let decodes = budget > 1
            && self
                .engine
                .max_context
                .is_none_or(|limit| prompt_len < limit);
        if !decodes {
            0
        } else if prompt_len.is_multiple_of(self.engine.block_size) {
            num_samples
        } else {
            num_samples - 1
        }
    }

    /// Blocks a prompt occupies once admitted, plus its first-step growth: what
    /// the pool must hold, in total, for the request to be admissible at all.
    fn first_step_blocks(&self, prompt_len: usize, num_samples: usize, budget: usize) -> usize {
        prompt_len.div_ceil(self.engine.block_size).max(1)
            + self.first_step_growth(prompt_len, num_samples, budget)
    }

    /// Blocks the live sequences will have to allocate at their next step: one
    /// for each whose held token lands past its mapping or in a block it does
    /// not own outright. Admission must leave at least this many free.
    fn next_step_reserve(&self) -> usize {
        let block_size = self.engine.block_size;
        self.active
            .iter()
            .filter(|s| s.finished.is_none() && !s.preempted)
            .map(|s| {
                let pos = s.token_ids.len() - 1;
                match s.block_table.get_physical_location(pos, block_size) {
                    Some((block, _)) => usize::from(self.kv.is_shared(block)),
                    None => 1,
                }
            })
            .sum()
    }

    // ── admission and prefill ────────────────────────────────────────────────

    /// Admit only the head request. A partial prompt keeps the lane across
    /// iterations; newer arrivals cannot continually overtake it.
    fn start_prefill(&mut self) -> bool {
        let Some(req) = self.waiting.front() else {
            return false;
        };
        let seq_id = req
            .sequence_ids
            .first()
            .copied()
            .or_else(|| req.resume.as_ref().map(|p| p.sequence_id))
            .unwrap_or(self.next_sequence_id);
        let budget = req.max_tokens - req.resume.as_ref().map_or(0, |p| p.generated.len());
        let growth = self.first_step_growth(req.tokens.len(), req.num_samples, budget);
        let reserved = self.next_step_reserve();
        let Some(admission) =
            self.kv
                .admit_with_headroom(seq_id, &req.tokens, self.tick, reserved + growth)
        else {
            return false;
        };
        let mut req = self.waiting.pop_front().expect("front was checked");
        let cursor = admission.cached_tokens.min(req.tokens.len() - 1);
        if req.sequence_ids.is_empty() {
            if let Some(p) = &req.resume {
                req.sequence_ids.push(p.sequence_id);
            } else {
                req.sequence_ids =
                    (self.next_sequence_id..self.next_sequence_id + req.num_samples).collect();
                self.next_sequence_id += req.num_samples;
            }
        }
        if req.admitted_at.is_none() {
            req.admitted_at = Some(Instant::now());
            if req.resume.is_none() {
                self.stats.requests += 1;
                self.stats.prompt_tokens += req.tokens.len();
                self.stats.prompt_tokens_reused += cursor;
            }
        }
        req.prefill_high_water = req.prefill_high_water.max(cursor);
        self.prefilling = Some(Prefilling {
            request: req,
            block_table: admission.block_table,
            cursor,
        });
        true
    }

    fn advance_prefill(&mut self) {
        let mut remaining = self.engine.max_prefill_tokens_per_step;
        while remaining > 0 {
            if self.prefilling.is_none() && !self.start_prefill() {
                break;
            }
            let mut prefill = self.prefilling.take().expect("prefill present");
            let end = prefill
                .cursor
                .saturating_add(remaining)
                .min(prefill.request.tokens.len());
            let count = end - prefill.cursor;
            let complete = end == prefill.request.tokens.len();
            let started = Instant::now();
            self.weights.prefill_range(
                &prefill.request.tokens[prefill.cursor..end],
                prefill.cursor,
                &self.config,
                &prefill.block_table,
                &mut self.kv_cache,
                self.engine.block_size,
                self.engine.prefill_chunk_size,
                &mut self.batch_scratch,
                complete,
            );
            self.stats.prefill_time += started.elapsed();
            self.stats.prefill_chunks += 1;
            self.stats.last_prefill_tokens += count;
            if prefill.request.resume.is_some() {
                self.stats.recomputed_tokens += count;
            } else {
                self.stats.prompt_tokens_prefilled += count;
                self.stats.recomputed_tokens += end
                    .min(prefill.request.prefill_high_water)
                    .saturating_sub(prefill.cursor);
            }
            prefill.request.prefill_high_water = prefill.request.prefill_high_water.max(end);
            prefill.cursor = end;
            remaining -= count;
            if complete {
                // Sample immediately: any later model call reuses the logits
                // scratch. Publication happens only after ALL prompt KV is valid.
                self.finish_prefill(prefill);
                self.reclaim();
            } else {
                self.prefilling = Some(prefill);
            }
        }
    }

    /// Decode gets first claim on memory as well as execution. A partial
    /// prompt is unpublished and can be replayed later without changing output.
    fn evict_prefill(&mut self) -> bool {
        let Some(prefill) = self.prefilling.take() else {
            return false;
        };
        self.kv.release_sequence(prefill.request.sequence_ids[0]);
        self.stats.prefill_preemptions += 1;
        self.waiting.push_front(prefill.request);
        true
    }

    fn finish_prefill(&mut self, prefill: Prefilling) {
        let req = prefill.request;
        let admitted_at = req.admitted_at.expect("admission timestamp retained");
        let vocab = self.config.vocab_size;
        self.kv.publish_prompt_blocks(&prefill.block_table);
        let mut tables = vec![(req.sequence_ids[0], prefill.block_table)];
        for &child in &req.sequence_ids[1..] {
            let table = self.kv.fork(&tables[0].1, child, self.tick);
            tables.push((child, table));
        }
        let Request {
            id: request_id,
            tokens: prompt,
            max_tokens,
            num_samples,
            options,
            submitted_at,
            resume,
            ..
        } = req;

        match resume {
            Some(p) => {
                // Continue where it left off: its held token was replayed
                // as the tail of the prompt, and its sampler picks up its
                // own stream.
                debug_assert_eq!(tables.len(), 1, "a resumed sequence has no siblings");
                let (sid, table) = tables.pop().expect("one table");
                let mut sampler = p.sampler;
                let next = sampler.sample(self.batch_scratch.logits_for_mut(0, vocab));
                self.stats.generated_tokens += 1;
                let mut generated = p.generated;
                generated.push(next);
                let mut token_ids = prompt;
                token_ids.push(next);
                let finished = stop_reason(
                    self.engine.eos_token,
                    &self.engine.extra_eos_tokens,
                    self.engine.max_context,
                    next,
                    generated.len(),
                    max_tokens,
                    token_ids.len(),
                );
                self.active.push(Sequence {
                    id: sid,
                    request_id,
                    prompt_len: p.prompt_len,
                    token_ids,
                    generated,
                    max_tokens,
                    block_table: table,
                    finished,
                    sampler,
                    submitted_at,
                    admitted_at: p.admitted_at,
                    first_token_at: p.first_token_at,
                    drafter: p.drafter,
                    deferred: false,
                    preempted: false,
                });
                self.record_delta(request_id, sid, vec![next], finished);
            }
            None => {
                for (i, (sid, table)) in tables.into_iter().enumerate() {
                    let (mut sampler, greedy) = self.sampler_for(&options, num_samples, sid, i);
                    let first = sampler.sample(self.batch_scratch.logits_for_mut(0, vocab));
                    self.stats.generated_tokens += 1;

                    let mut token_ids = prompt.clone();
                    token_ids.push(first);
                    let finished = stop_reason(
                        self.engine.eos_token,
                        &self.engine.extra_eos_tokens,
                        self.engine.max_context,
                        first,
                        1,
                        max_tokens,
                        token_ids.len(),
                    );
                    let first_token_at = Some(Instant::now());

                    self.active.push(Sequence {
                        id: sid,
                        request_id,
                        prompt_len: prompt.len(),
                        token_ids,
                        generated: vec![first],
                        max_tokens,
                        block_table: table,
                        finished,
                        sampler,
                        submitted_at,
                        admitted_at,
                        first_token_at,
                        // Speculation is only sound under greedy; see
                        // EngineConfig.
                        drafter: (self.engine.draft_tokens > 0 && greedy)
                            .then(|| Box::new(PromptLookupDrafter::default()) as Box<dyn Drafter>),
                        deferred: false,
                        preempted: false,
                    });
                    // The token sampled off the prefill logits is a real
                    // emission — it is the one a streaming client is
                    // waiting on.
                    self.record_delta(request_id, sid, vec![first], finished);
                }
            }
        }
    }

    /// The sampler for sample `i` of a request, and whether it is greedy.
    fn sampler_for(
        &self,
        options: &RequestOptions,
        num_samples: usize,
        sequence_id: usize,
        i: usize,
    ) -> (Sampler, bool) {
        let requested = options.temperature;
        let mut temperature = requested.unwrap_or(self.engine.temperature);
        // Greedy siblings would all replay the same continuation, so a
        // multi-sample request that did not choose a temperature gets a
        // non-zero one. One that chose 0.0 asked for identical branches and
        // gets them.
        if num_samples > 1 && requested.is_none() && temperature <= 0.0 {
            temperature = 0.8;
        }
        let top_p = options.top_p.unwrap_or(self.engine.top_p);
        let top_k = options.top_k.unwrap_or(self.engine.top_k);
        let seed = match options.seed {
            // A request that brings its own seed must replay exactly on its
            // own, whatever else the engine ran before it: derive from the
            // sample index, not from the global sequence counter.
            Some(seed) => seed ^ i as u64,
            None => self.engine.seed ^ ((sequence_id as u64) << 16) ^ i as u64,
        };
        (
            Sampler::new(temperature, top_p, top_k, seed),
            temperature <= 0.0,
        )
    }

    // ── decode ───────────────────────────────────────────────────────────────

    /// Advance every live sequence, by one token or by several.
    ///
    /// Every sequence contributes a *group* of positions to one batch: the
    /// token it already holds, plus however many tokens its drafter guessed. A
    /// sequence that is not speculating contributes a group of one, so ordinary
    /// decoding is just the K=0 case of this path rather than a separate one.
    ///
    /// Memory is settled for all of them first, because the batched forward
    /// needs the block tables final and immutable while growing a mapping or
    /// splitting a shared block mutates them.
    fn decode(&mut self) {
        let t0 = Instant::now();
        let groups = self.plan_step();
        if !groups.is_empty() {
            let predictions = self.run_groups(&groups);
            self.commit(&groups, &predictions);
        }
        self.stats.decode_time += t0.elapsed();
    }

    /// Settle memory for the step: every mandatory write first, then whatever
    /// optional speculation the remaining headroom allows.
    ///
    /// Returns the groups that will run. A sequence missing from the result
    /// either finished, is deferred until memory frees up, or was preempted.
    fn plan_step(&mut self) -> Vec<SpecGroup> {
        // Pass 1: the held token of every live sequence. Nothing optional is
        // touched until all of these are settled, so a draft can never take
        // the block a sibling needs to advance at all.
        let mut runnable = Vec::with_capacity(self.active.len());
        let mut deferred = Vec::new();
        for idx in 0..self.active.len() {
            if self.active[idx].finished.is_some() || self.active[idx].preempted {
                continue;
            }
            if self.secure_held_token(idx) || (self.evict_prefill() && self.secure_held_token(idx))
            {
                runnable.push(idx);
            } else {
                deferred.push(idx);
            }
        }

        // Deadlock: something is waiting for memory and nothing can move, so
        // no future step would free anything either. Preempt the newest
        // waiter — least work to redo — and retry the rest with its blocks.
        while runnable.is_empty() && !deferred.is_empty() {
            let newest = (0..deferred.len())
                .max_by_key(|&i| self.active[deferred[i]].id)
                .expect("non-empty");
            let victim = deferred.swap_remove(newest);
            self.preempt(victim);
            for idx in std::mem::take(&mut deferred) {
                if self.secure_held_token(idx) {
                    runnable.push(idx);
                } else {
                    deferred.push(idx);
                }
            }
        }
        self.stats.deferred_steps += deferred.len();

        // Pass 2: optional speculation, allowed only while a block per live
        // sequence stays free for the *next* step's mandatory writes.
        let live = runnable.len() + deferred.len();
        let mut groups = Vec::with_capacity(runnable.len());
        for idx in runnable {
            let drafts = self.plan_drafts(idx, live);
            let held = *self.active[idx].token_ids.last().expect("never empty");
            let pos = self.active[idx].token_ids.len() - 1;
            let mut tokens = Vec::with_capacity(drafts.len() + 1);
            tokens.push(held);
            tokens.extend_from_slice(&drafts);
            let positions = (pos..pos + tokens.len()).collect();
            groups.push(SpecGroup {
                seq_idx: idx,
                tokens,
                positions,
                drafts,
            });
        }
        groups
    }

    /// Map and privately own the slot for the token a sequence already holds.
    /// Returns false — and marks the sequence deferred — if that is not
    /// possible right now, in which case nothing was written and the sequence
    /// still holds exactly the blocks it had.
    fn secure_held_token(&mut self, idx: usize) -> bool {
        let block_size = self.engine.block_size;
        let seq_id = self.active[idx].id;
        let pos = self.active[idx].token_ids.len() - 1;
        let mut table = std::mem::take(&mut self.active[idx].block_table);

        let mut secured = true;
        while pos >= table.len() * block_size {
            if !self.kv.append_block(seq_id, &mut table, self.tick) {
                secured = false;
                break;
            }
        }
        if secured {
            // Forked siblings map the same partial block; split before
            // writing. A block that could not be split must not be written.
            secured = self
                .kv
                .ensure_writable(seq_id, &mut table, pos, &mut self.kv_cache, &self.layout)
                .is_writable();
        }
        self.active[idx].block_table = table;
        self.active[idx].deferred = !secured;
        if secured {
            self.kv.touch(seq_id, self.tick);
        }
        secured
    }

    /// Ask the drafter for guesses and map slots for as many of them as the
    /// pool can spare. Drafts are optional: any that cannot be housed are
    /// simply dropped, never deferred for.
    fn plan_drafts(&mut self, idx: usize, live: usize) -> Vec<u32> {
        let k = self.engine.draft_tokens;
        if k == 0 || self.active[idx].drafter.is_none() {
            return Vec::new();
        }
        let block_size = self.engine.block_size;
        let seq_id = self.active[idx].id;
        let pos = self.active[idx].token_ids.len() - 1;

        // Never draft past the request's token budget or the model's context:
        // a step emits one more token than it accepts.
        let budget = self.active[idx]
            .max_tokens
            .saturating_sub(self.active[idx].generated.len());
        let context_room = self
            .engine
            .max_context
            .map_or(usize::MAX, |limit| limit.saturating_sub(pos + 1));
        let room = budget.saturating_sub(1).min(context_room);
        if room == 0 {
            return Vec::new();
        }

        // The drafter is moved out so it can be given the sequence's own token
        // history without aliasing it.
        let mut drafter = self.active[idx].drafter.take().expect("checked above");
        let mut drafts = drafter.draft(&self.active[idx].token_ids, k.min(room));
        self.active[idx].drafter = Some(drafter);
        drafts.truncate(room);

        let mut table = std::mem::take(&mut self.active[idx].block_table);
        let mut housed = 0;
        for j in 0..drafts.len() {
            let p = pos + 1 + j;
            // Optional work may allocate only while one block per live
            // sequence stays free for next step's mandatory writes.
            let may_allocate = self.kv.available_blocks() > live;
            if p >= table.len() * block_size
                && (!may_allocate || !self.kv.append_block(seq_id, &mut table, self.tick))
            {
                break;
            }
            // A draft slot is normally private already — it sits in the held
            // token's block, split in pass 1, or in a block appended just now.
            // Should it ever need a copy, that copy is optional too.
            if !may_allocate && self.kv.is_shared_position(&table, p) {
                break;
            }
            if !self
                .kv
                .ensure_writable(seq_id, &mut table, p, &mut self.kv_cache, &self.layout)
                .is_writable()
            {
                break;
            }
            housed = j + 1;
        }
        self.active[idx].block_table = table;
        drafts.truncate(housed);
        drafts
    }

    /// Take a sequence's blocks away so others can move. `reclaim` returns it
    /// to the head of the queue to be recomputed, or completes it if it could
    /// never be re-admitted.
    fn preempt(&mut self, idx: usize) {
        let seq_id = self.active[idx].id;
        self.kv.release_sequence(seq_id);
        self.active[idx].block_table = BlockTable::new();
        self.active[idx].deferred = false;
        self.active[idx].preempted = true;
        self.stats.preemptions += 1;
    }

    /// Run every group's positions through the model and return, per group, the
    /// token the model itself chose at each position.
    ///
    /// Groups are flattened into one flat batch so the weights stream once for
    /// all of them. Splitting a group across chunks is safe: a chunk runs every
    /// layer before the next begins, so a later position always finds the KV of
    /// the earlier ones already written.
    fn run_groups(&mut self, groups: &[SpecGroup]) -> Vec<Vec<u32>> {
        let vocab = self.config.vocab_size;
        let block_size = self.engine.block_size;

        let mut flat_tokens = Vec::new();
        let mut flat_positions = Vec::new();
        let mut owner = Vec::new();
        for (g, group) in groups.iter().enumerate() {
            for (t, p) in group.tokens.iter().zip(group.positions.iter()) {
                flat_tokens.push(*t);
                flat_positions.push(*p);
                owner.push(g);
            }
        }

        let mut predictions: Vec<Vec<u32>> = groups
            .iter()
            .map(|g| Vec::with_capacity(g.tokens.len()))
            .collect();

        let capacity = self.batch_scratch.capacity();
        let mut start = 0;
        while start < flat_tokens.len() {
            let end = (start + capacity).min(flat_tokens.len());
            let tables: Vec<&BlockTable> = owner[start..end]
                .iter()
                .map(|&g| &self.active[groups[g].seq_idx].block_table)
                .collect();

            self.weights.decode_batch_into(
                &flat_tokens[start..end],
                &flat_positions[start..end],
                &tables,
                &self.config,
                &mut self.kv_cache,
                block_size,
                &mut self.batch_scratch,
            );
            drop(tables);

            for b in 0..(end - start) {
                let g = owner[start + b];
                let idx = groups[g].seq_idx;
                let logits = self.batch_scratch.logits_for_mut(b, vocab);
                // Speculating sequences are greedy by construction, so argmax is
                // both the verification rule and what their sampler would return.
                // Everything else goes through its own sampler.
                let token = if self.active[idx].drafter.is_some() {
                    crate::sampling::argmax(logits) as u32
                } else {
                    self.active[idx].sampler.sample(logits)
                };
                predictions[g].push(token);
            }

            self.spec.passes += 1;
            start = end;
        }
        predictions
    }

    /// Accept the drafts the model agreed with and append the result.
    fn commit(&mut self, groups: &[SpecGroup], predictions: &[Vec<u32>]) {
        let eos = self.engine.eos_token;
        let extra_eos = std::mem::take(&mut self.engine.extra_eos_tokens);
        let max_context = self.engine.max_context;

        for (g, group) in groups.iter().enumerate() {
            let verdict = verify_greedy(&group.drafts, &predictions[g]);
            let idx = group.seq_idx;

            self.spec.steps += 1;
            self.spec.drafted += verdict.drafted as u64;
            self.spec.accepted += verdict.accepted.len() as u64;

            let mut emitted = Vec::with_capacity(verdict.accepted.len() + 1);
            for token in verdict
                .accepted
                .iter()
                .copied()
                .chain(std::iter::once(verdict.corrected))
            {
                let seq = &mut self.active[idx];
                if seq.finished.is_some() {
                    break;
                }
                seq.generated.push(token);
                seq.token_ids.push(token);
                emitted.push(token);
                seq.finished = stop_reason(
                    eos,
                    &extra_eos,
                    max_context,
                    token,
                    seq.generated.len(),
                    seq.max_tokens,
                    seq.token_ids.len(),
                );
            }

            // KV written for rejected drafts is left in place. It is never read:
            // attention only ever looks at positions below the sequence's
            // length, and the next step overwrites those slots.
            self.spec.emitted += emitted.len() as u64;
            self.stats.generated_tokens += emitted.len();

            let seq = &self.active[idx];
            let (request_id, sequence_id, finished) = (seq.request_id, seq.id, seq.finished);
            self.record_delta(request_id, sequence_id, emitted, finished);
        }
        self.engine.extra_eos_tokens = extra_eos;
    }

    // ── reclamation ──────────────────────────────────────────────────────────

    /// Free finished sequences immediately, so the next `step()` can admit
    /// against the memory they held, and send preempted ones back to the
    /// queue. Blocks the prefix cache still references stay resident.
    fn reclaim(&mut self) {
        let mut done = Vec::new();
        let mut preempted = Vec::new();
        for seq in std::mem::take(&mut self.active) {
            if seq.finished.is_some() {
                done.push(seq);
            } else if seq.preempted {
                preempted.push(seq);
            } else {
                self.active.push(seq);
            }
        }

        for seq in done {
            self.kv.release_sequence(seq.id);
            let reason = seq.finished.expect("finished");
            self.completed.push(Self::completion_of(&seq, reason));
        }

        // Newest first, each to the front: the oldest ends up at the head, so
        // resumption is in admission order.
        preempted.sort_by_key(|s| std::cmp::Reverse(s.id));
        for seq in preempted {
            self.requeue(seq);
        }
    }

    fn completion_of(seq: &Sequence, reason: FinishReason) -> Completion {
        let first = seq.first_token_at.unwrap_or(seq.admitted_at);
        Completion {
            request_id: seq.request_id,
            sequence_id: seq.id,
            prompt_tokens: seq.prompt_len,
            tokens: seq.generated.clone(),
            finish_reason: reason,
            time_to_first_token: first.duration_since(seq.submitted_at),
            queue_time: seq.admitted_at.duration_since(seq.submitted_at),
        }
    }

    /// Put a preempted sequence back at the head of the queue, or complete it
    /// with `OutOfMemory` if not even an empty pool could take it back.
    fn requeue(&mut self, seq: Sequence) {
        let budget = seq.max_tokens - seq.generated.len();
        let needed = self.first_step_blocks(seq.token_ids.len(), 1, budget);
        if needed > self.engine.total_blocks {
            let reason = FinishReason::OutOfMemory;
            self.record_delta(seq.request_id, seq.id, Vec::new(), Some(reason));
            self.completed.push(Self::completion_of(&seq, reason));
            return;
        }
        let Sequence {
            id,
            request_id,
            prompt_len,
            token_ids,
            generated,
            max_tokens,
            sampler,
            submitted_at,
            admitted_at,
            first_token_at,
            drafter,
            ..
        } = seq;
        self.waiting.push_front(Request {
            id: request_id,
            tokens: token_ids,
            max_tokens,
            num_samples: 1,
            options: RequestOptions::default(),
            submitted_at,
            sequence_ids: vec![id],
            admitted_at: Some(admitted_at),
            prefill_high_water: 0,
            resume: Some(Preempted {
                sequence_id: id,
                prompt_len,
                generated,
                sampler,
                drafter,
                admitted_at,
                first_token_at,
            }),
        });
    }
}
