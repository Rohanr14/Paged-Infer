//! The serving loop: iteration-level scheduling over a paged KV cache.
//!
//! One `step()` admits whatever fits, prefills it, advances every live sequence
//! by one token, and reclaims what finished — so a request that completes frees
//! its blocks for the next admission immediately, rather than at the end of some
//! fixed batch. That is what continuous batching means in practice.
//!
//! Deliberately independent of tokenization: the scheduler works on token ids,
//! and text is a thin convenience layer on top. That keeps the whole engine
//! testable against a synthetic checkpoint without dragging in a real tokenizer.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::memory::block_table::BlockTable;
use crate::memory::kv_cache_manager::KvCacheManager;
use crate::memory::layout::KvLayout;
use crate::memory::prefix_cache::PrefixCacheStats;
use crate::model::{ForwardScratch, LlamaConfig, LlamaWeights};
use crate::sampling::Sampler;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub total_blocks: usize,
    pub block_size: usize,
    /// `0.0` is greedy. Forked samples override this upward, since identical
    /// greedy branches would make forking pointless.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub eos_token: u32,
    /// Prepended to every prompt when set.
    pub bos_token: Option<u32>,
    pub enable_prefix_cache: bool,
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
            bos_token: Some(1),
            enable_prefix_cache: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted the end-of-sequence token.
    Eos,
    /// The request's token budget ran out.
    Length,
    /// The KV cache could not grow to hold another token.
    OutOfMemory,
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub request_id: usize,
    pub sequence_id: usize,
    pub prompt_tokens: usize,
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    /// Admission through to the first sampled token — the prefill cost this
    /// request actually paid, after prefix reuse.
    pub time_to_first_token: Duration,
}

#[derive(Debug, Default, Clone)]
pub struct RunStats {
    pub requests: usize,
    pub prompt_tokens: usize,
    /// Prompt tokens that went through the model. The gap against
    /// `prompt_tokens` is what the prefix cache saved.
    pub prompt_tokens_prefilled: usize,
    pub generated_tokens: usize,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub steps: usize,
}

impl RunStats {
    pub fn prompt_tokens_reused(&self) -> usize {
        self.prompt_tokens - self.prompt_tokens_prefilled
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        self.generated_tokens as f64 / self.decode_time.as_secs_f64().max(1e-9)
    }
}

struct Request {
    id: usize,
    tokens: Vec<u32>,
    max_tokens: usize,
    num_samples: usize,
    /// Overrides the engine default when set.
    temperature: Option<f32>,
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
    admitted_at: Instant,
    first_token_at: Option<Instant>,
}

pub struct Engine<'a> {
    weights: LlamaWeights<'a>,
    config: LlamaConfig,
    engine: EngineConfig,
    kv: KvCacheManager,
    kv_cache: Vec<f32>,
    layout: KvLayout,
    scratch: ForwardScratch,
    tokenizer: Option<Tokenizer>,
    waiting: VecDeque<Request>,
    active: Vec<Sequence>,
    completed: Vec<Completion>,
    next_request_id: usize,
    next_sequence_id: usize,
    tick: u64,
    stats: RunStats,
}

impl<'a> Engine<'a> {
    pub fn new(weights: LlamaWeights<'a>, config: LlamaConfig, engine: EngineConfig) -> Self {
        let layout = config.kv_layout(engine.total_blocks, engine.block_size);
        let kv_cache = vec![0.0; layout.total_floats()];
        let kv = KvCacheManager::new(engine.total_blocks, engine.block_size)
            .with_prefix_cache(engine.enable_prefix_cache);
        let scratch = ForwardScratch::new(&config);

        Self {
            weights,
            config,
            engine,
            kv,
            kv_cache,
            layout,
            scratch,
            tokenizer: None,
            waiting: VecDeque::new(),
            active: Vec::new(),
            completed: Vec::new(),
            next_request_id: 0,
            next_sequence_id: 0,
            tick: 0,
            stats: RunStats::default(),
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

    /// Queue a prompt that is already tokenized. `num_samples` continuations are
    /// drawn from it; the prompt is prefilled once regardless.
    pub fn submit_tokens(
        &mut self,
        tokens: Vec<u32>,
        max_tokens: usize,
        num_samples: usize,
    ) -> usize {
        self.submit_tokens_with(tokens, max_tokens, num_samples, None)
    }

    /// [`Engine::submit_tokens`] with a per-request sampling temperature.
    /// `None` uses the engine default. A server needs this: temperature is a
    /// property of the request, not of the engine.
    pub fn submit_tokens_with(
        &mut self,
        tokens: Vec<u32>,
        max_tokens: usize,
        num_samples: usize,
        temperature: Option<f32>,
    ) -> usize {
        assert!(!tokens.is_empty(), "cannot generate from an empty prompt");
        let id = self.next_request_id;
        self.next_request_id += 1;
        let mut tokens = tokens;
        if let Some(bos) = self.engine.bos_token {
            tokens.insert(0, bos);
        }
        self.waiting.push_back(Request {
            id,
            tokens,
            max_tokens,
            num_samples: num_samples.max(1),
            temperature,
        });
        id
    }

    /// Queue a text prompt. Requires a tokenizer.
    pub fn submit(&mut self, prompt: &str, max_tokens: usize, num_samples: usize) -> Result<usize> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("engine has no tokenizer; use submit_tokens"))?;
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        Ok(self.submit_tokens(encoding.get_ids().to_vec(), max_tokens, num_samples))
    }

    pub fn decode_text(&self, tokens: &[u32]) -> Option<String> {
        self.tokenizer.as_ref()?.decode(tokens, true).ok()
    }

    /// True while anything is queued or running.
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.active.is_empty()
    }

    /// Take the completions finished since the last call.
    ///
    /// This is what lets a server interleave: submit whenever a client
    /// connects, `step()` continuously, and hand back each completion as it
    /// lands, rather than waiting for the whole batch to drain.
    pub fn take_completed(&mut self) -> Vec<Completion> {
        std::mem::take(&mut self.completed)
    }

    /// Drain both queues, returning every completion in finish order.
    pub fn run(&mut self) -> Result<Vec<Completion>> {
        while !self.waiting.is_empty() || !self.active.is_empty() {
            let before = self.waiting.len() + self.active.len();
            self.step()?;
            if self.waiting.len() + self.active.len() == before && self.active.is_empty() {
                // Nothing running and nothing admissible: the smallest waiting
                // prompt cannot fit in the whole cache, so it never will.
                anyhow::bail!(
                    "a queued prompt does not fit in {} blocks of {} tokens",
                    self.engine.total_blocks,
                    self.engine.block_size
                );
            }
        }
        Ok(std::mem::take(&mut self.completed))
    }

    /// Admit, decode one token per live sequence, reclaim.
    pub fn step(&mut self) -> Result<()> {
        self.tick += 1;
        self.admit();
        self.decode();
        self.reclaim();
        self.stats.steps += 1;
        Ok(())
    }

    // ── admission and prefill ────────────────────────────────────────────────

    fn admit(&mut self) {
        while let Some(req) = self.waiting.front() {
            let seq_id = self.next_sequence_id;
            let Some(admission) = self.kv.admit(seq_id, &req.tokens, self.tick) else {
                // Out of memory even after reclaiming cold cache blocks. Wait
                // for a live sequence to retire.
                break;
            };
            let req = self.waiting.pop_front().expect("front was just checked");
            self.next_sequence_id += 1;
            let admitted_at = Instant::now();

            // Replay only what the cache did not cover. If the whole prompt was
            // cached we still need logits, so the last token is recomputed --
            // its KV is rewritten with identical values.
            let resume_at = admission.cached_tokens.min(req.tokens.len() - 1);
            let t0 = Instant::now();
            self.weights.prefill_into(
                &req.tokens[resume_at..],
                resume_at,
                &self.config,
                &admission.block_table,
                &mut self.kv_cache,
                self.engine.block_size,
                None,
                &mut self.scratch,
            );
            self.stats.prefill_time += t0.elapsed();
            self.stats.requests += 1;
            self.stats.prompt_tokens += req.tokens.len();
            self.stats.prompt_tokens_prefilled += req.tokens.len() - resume_at;

            // The prompt's blocks now hold real KV state; offer them up.
            self.kv.publish_prompt_blocks(&admission.block_table);

            // Extra samples map the parent's blocks rather than re-prefilling.
            // Nothing is copied here — divergence is paid for lazily, per block.
            let mut tables = vec![(seq_id, admission.block_table)];
            for _ in 1..req.num_samples {
                let child = self.next_sequence_id;
                self.next_sequence_id += 1;
                let table = self.kv.fork(&tables[0].1, child, self.tick);
                tables.push((child, table));
            }

            for (i, (sid, table)) in tables.into_iter().enumerate() {
                // Siblings need distinct seeds, and greedy siblings would all
                // replay the same continuation, so multi-sample requests get a
                // non-zero temperature by default.
                let requested = req.temperature.unwrap_or(self.engine.temperature);
                let temperature = if req.num_samples > 1 && requested <= 0.0 {
                    0.8
                } else {
                    requested
                };
                let mut sampler = Sampler::new(
                    temperature,
                    self.engine.top_p,
                    self.engine.top_k,
                    self.engine.seed ^ ((sid as u64) << 16) ^ i as u64,
                );
                let first = sampler.sample(&mut self.scratch.logits);
                self.stats.generated_tokens += 1;

                let mut token_ids = req.tokens.clone();
                token_ids.push(first);
                let finished = (first == self.engine.eos_token)
                    .then_some(FinishReason::Eos)
                    .or((req.max_tokens <= 1).then_some(FinishReason::Length));

                self.active.push(Sequence {
                    id: sid,
                    request_id: req.id,
                    prompt_len: req.tokens.len(),
                    token_ids,
                    generated: vec![first],
                    max_tokens: req.max_tokens,
                    block_table: table,
                    finished,
                    sampler,
                    admitted_at,
                    first_token_at: Some(Instant::now()),
                });
            }
        }
    }

    // ── decode ───────────────────────────────────────────────────────────────

    fn decode(&mut self) {
        let block_size = self.engine.block_size;
        let t0 = Instant::now();

        for idx in 0..self.active.len() {
            if self.active[idx].finished.is_some() {
                continue;
            }

            let pos = self.active[idx].token_ids.len() - 1;
            let seq_id = self.active[idx].id;

            // Grow the mapping if this position runs past the last block.
            if pos >= self.active[idx].block_table.len() * block_size {
                let mut table = std::mem::take(&mut self.active[idx].block_table);
                let grew = self.kv.append_block(seq_id, &mut table, self.tick);
                self.active[idx].block_table = table;
                if !grew {
                    self.active[idx].finished = Some(FinishReason::OutOfMemory);
                    continue;
                }
            }

            // Forked siblings map the same partial block. Split it before
            // writing so one sample cannot corrupt another's KV.
            let mut table = std::mem::take(&mut self.active[idx].block_table);
            self.kv
                .ensure_writable(seq_id, &mut table, pos, &mut self.kv_cache, &self.layout);
            self.active[idx].block_table = table;

            let token = *self.active[idx].token_ids.last().expect("never empty");
            self.weights.forward_into(
                token,
                pos,
                &self.config,
                &self.active[idx].block_table,
                &mut self.kv_cache,
                block_size,
                None,
                &mut self.scratch,
            );

            let seq = &mut self.active[idx];
            let next = seq.sampler.sample(&mut self.scratch.logits);
            seq.generated.push(next);
            seq.token_ids.push(next);
            self.stats.generated_tokens += 1;

            if next == self.engine.eos_token {
                seq.finished = Some(FinishReason::Eos);
            } else if seq.generated.len() >= seq.max_tokens {
                seq.finished = Some(FinishReason::Length);
            }
            self.kv.touch(seq_id, self.tick);
        }

        self.stats.decode_time += t0.elapsed();
    }

    // ── reclamation ──────────────────────────────────────────────────────────

    /// Free finished sequences immediately, so the next `step()` can admit
    /// against the memory they held. Blocks the prefix cache still references
    /// stay resident.
    fn reclaim(&mut self) {
        let mut done = Vec::new();
        self.active.retain(|seq| match seq.finished {
            Some(reason) => {
                done.push((
                    seq.id,
                    Completion {
                        request_id: seq.request_id,
                        sequence_id: seq.id,
                        prompt_tokens: seq.prompt_len,
                        tokens: seq.generated.clone(),
                        finish_reason: reason,
                        time_to_first_token: seq
                            .first_token_at
                            .unwrap_or(seq.admitted_at)
                            .duration_since(seq.admitted_at),
                    },
                ));
                false
            }
            None => true,
        });

        for (seq_id, completion) in done {
            self.kv.release_sequence(seq_id);
            self.completed.push(completion);
        }
    }
}
