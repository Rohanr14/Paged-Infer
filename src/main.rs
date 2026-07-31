use anyhow::Result;
use memmap2::MmapOptions;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::memory::kv_cache_manager::KvCacheManager;
use paged_infer::memory::layout::KvLayout;
use paged_infer::model::{ForwardScratch, LlamaConfig, LlamaWeights, ModelLoader};
use paged_infer::sampling::Sampler;
use std::collections::VecDeque;
use std::fs::File;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

// TinyLlama (Llama 2 architecture) special tokens
const BOS_TOKEN: u32 = 1; // <s>
const EOS_TOKEN: u32 = 2; // </s>

/// Represents an incoming user request.
pub struct Request {
    pub id: usize,
    pub prompt: String,
    pub max_tokens: usize,
    /// Independent continuations to draw from this one prompt. Samples beyond
    /// the first are forked, sharing the prompt's KV blocks copy-on-write.
    pub num_samples: usize,
}

/// Represents an active generation sequence within the engine.
pub struct Sequence {
    pub id: usize,
    /// The request this sequence samples from; siblings share it.
    pub request_id: usize,
    pub token_ids: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub block_table: BlockTable,
    pub is_finished: bool,
    pub sampler: Sampler,
}

/// What one engine run cost.
#[derive(Debug, Default, Clone)]
pub struct RunStats {
    pub prompt_tokens: usize,
    pub prompt_tokens_prefilled: usize,
    pub generated_tokens: usize,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub steps: usize,
}

pub struct Engine<'a> {
    kv: KvCacheManager,
    tokenizer: Tokenizer,
    waiting_queue: VecDeque<Request>,
    active_batch: Vec<Sequence>,
    next_request_id: usize,
    next_sequence_id: usize,
    weights: LlamaWeights<'a>,
    config: LlamaConfig,
    kv_cache: Vec<f32>, // <--- The Actual Physical Memory Pool
    layout: KvLayout,
    scratch: ForwardScratch,
    tick: u64,
    temperature: f32,
    stats: RunStats,
}

impl<'a> Engine<'a> {
    pub fn new(
        tokenizer_path: &str,
        weights: LlamaWeights<'a>,
        config: LlamaConfig,
        total_blocks: usize,
        block_size: usize,
    ) -> Result<Self> {
        println!("Initializing Paged-Infer Engine...");
        let kv = KvCacheManager::new(total_blocks, block_size);
        println!("Allocated {total_blocks} physical blocks ({block_size} tokens per block).");

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        println!("Tokenizer loaded successfully.");

        let layout = config.kv_layout(total_blocks, block_size);
        let kv_cache = vec![0.0; layout.total_floats()];
        println!(
            "Allocated {:.2} MB for the Physical KV Cache.",
            (layout.total_floats() * 4) as f32 / 1_048_576.0
        );

        let scratch = ForwardScratch::new(&config);

        Ok(Self {
            kv,
            tokenizer,
            waiting_queue: VecDeque::new(),
            active_batch: Vec::new(),
            next_request_id: 0,
            next_sequence_id: 0,
            weights,
            config,
            kv_cache,
            layout,
            scratch,
            tick: 0,
            temperature: 0.0,
            stats: RunStats::default(),
        })
    }

    /// Sampling temperature for every sequence. Zero is greedy.
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    pub fn stats(&self) -> &RunStats {
        &self.stats
    }

    /// Submits a new prompt to the engine's queue.
    pub fn add_request(&mut self, prompt: &str, max_tokens: usize) {
        self.add_request_n(prompt, max_tokens, 1);
    }

    /// Submit a prompt asking for `num_samples` independent continuations. The
    /// prompt is prefilled once no matter how many samples are drawn.
    pub fn add_request_n(&mut self, prompt: &str, max_tokens: usize, num_samples: usize) {
        self.waiting_queue.push_back(Request {
            id: self.next_request_id,
            prompt: prompt.to_string(),
            max_tokens,
            num_samples: num_samples.max(1),
        });
        self.next_request_id += 1;
    }

    /// Executes a single generation step across the active batch and manages the queue.
    pub fn step(&mut self) -> Result<()> {
        self.tick += 1;
        self.admit_waiting()?;
        self.decode_active();
        self.reclaim_finished();
        self.stats.steps += 1;
        Ok(())
    }

    // ── phase 1: scheduling and prefill ──────────────────────────────────────

    fn admit_waiting(&mut self) -> Result<()> {
        while let Some(req) = self.waiting_queue.front() {
            let encoding = self
                .tokenizer
                .encode(req.prompt.clone(), true)
                .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;

            // Prepend the Begin-Of-Sequence token
            let mut input_ids = vec![BOS_TOKEN];
            input_ids.extend_from_slice(encoding.get_ids());

            let seq_id = self.next_sequence_id;
            let Some(admission) = self.kv.admit(seq_id, &input_ids, self.tick) else {
                // Not enough physical memory even after reclaiming cold cache
                // blocks; wait for an active sequence to finish.
                break;
            };

            let req = self.waiting_queue.pop_front().expect("front just checked");
            self.next_sequence_id += 1;
            let mut block_table = admission.block_table;

            // Replay only the part of the prompt the cache did not cover. When
            // the whole prompt is cached we still need logits, so recompute the
            // final token -- its KV is rewritten with identical values.
            let resume_at = admission.cached_tokens.min(input_ids.len() - 1);
            let t0 = Instant::now();
            self.weights.prefill_into(
                &input_ids[resume_at..],
                resume_at,
                &self.config,
                &block_table,
                &mut self.kv_cache,
                self.kv.block_size(),
                None,
                &mut self.scratch,
            );
            self.stats.prefill_time += t0.elapsed();
            self.stats.prompt_tokens += input_ids.len();
            self.stats.prompt_tokens_prefilled += input_ids.len() - resume_at;

            // Now that the prompt's blocks hold real KV state, offer them up.
            self.kv.publish_prompt_blocks(&block_table);

            println!(
                "[Req {}] Scheduled. Prompt: {} tokens, {} reused from prefix cache ({} blocks), {} prefilled{}",
                req.id,
                input_ids.len(),
                admission.cached_tokens,
                admission.reused_blocks,
                input_ids.len() - resume_at,
                if req.num_samples > 1 {
                    format!(", {} samples", req.num_samples)
                } else {
                    String::new()
                },
            );

            // Additional samples fork the prompt instead of re-prefilling it.
            // No KV is copied here; divergence is paid for later, per block, and
            // only if the sequences actually write to the same one.
            let mut tables = Vec::with_capacity(req.num_samples);
            for _ in 1..req.num_samples {
                let child_id = self.next_sequence_id;
                self.next_sequence_id += 1;
                tables.push((child_id, self.kv.fork(&block_table, child_id, self.tick)));
            }
            tables.insert(0, (seq_id, std::mem::take(&mut block_table)));

            for (i, (sid, table)) in tables.into_iter().enumerate() {
                // Siblings need different seeds or they would all replay the
                // same continuation and the fork would buy nothing.
                let temperature = if req.num_samples > 1 && self.temperature <= 0.0 {
                    0.8
                } else {
                    self.temperature
                };
                let mut sampler = Sampler::new(temperature, 0.95, 0, (sid as u64) << 8 | i as u64);
                let first = sampler.sample(&mut self.scratch.logits);

                let mut token_ids = input_ids.clone();
                token_ids.push(first);
                self.stats.generated_tokens += 1;

                self.active_batch.push(Sequence {
                    id: sid,
                    request_id: req.id,
                    token_ids,
                    generated_tokens: vec![first],
                    max_tokens: req.max_tokens,
                    block_table: table,
                    is_finished: first == EOS_TOKEN,
                    sampler,
                });
            }
        }
        Ok(())
    }

    // ── phase 2: decode ──────────────────────────────────────────────────────

    fn decode_active(&mut self) {
        let block_size = self.kv.block_size();
        let t0 = Instant::now();

        for idx in 0..self.active_batch.len() {
            if self.active_batch[idx].is_finished {
                continue;
            }

            // Position of the token we are about to consume.
            let pos = self.active_batch[idx].token_ids.len() - 1;
            let seq_id = self.active_batch[idx].id;

            // Grow the mapping if this position runs past the last block.
            if pos >= self.active_batch[idx].block_table.len() * block_size {
                let mut table = std::mem::take(&mut self.active_batch[idx].block_table);
                let ok = self.kv.append_block(seq_id, &mut table, self.tick);
                self.active_batch[idx].block_table = table;
                if !ok {
                    println!("[Seq {seq_id}] Out of KV Cache! Forcing early termination.");
                    self.active_batch[idx].is_finished = true;
                    continue;
                }
            }

            // Forked siblings map the same partial block. Split it before
            // writing, so one sample cannot corrupt another's KV state.
            let mut table = std::mem::take(&mut self.active_batch[idx].block_table);
            self.kv
                .ensure_writable(seq_id, &mut table, pos, &mut self.kv_cache, &self.layout);
            self.active_batch[idx].block_table = table;

            let current_token = *self.active_batch[idx].token_ids.last().unwrap();
            self.weights.forward_into(
                current_token,
                pos,
                &self.config,
                &self.active_batch[idx].block_table,
                &mut self.kv_cache,
                block_size,
                None,
                &mut self.scratch,
            );

            let seq = &mut self.active_batch[idx];
            let next_token = seq.sampler.sample(&mut self.scratch.logits);
            seq.generated_tokens.push(next_token);
            seq.token_ids.push(next_token);
            self.stats.generated_tokens += 1;

            if next_token == EOS_TOKEN || seq.generated_tokens.len() >= seq.max_tokens {
                seq.is_finished = true;
            }
            self.kv.touch(seq_id, self.tick);
        }

        self.stats.decode_time += t0.elapsed();
    }

    // ── phase 3: cleanup ─────────────────────────────────────────────────────

    /// Free memory for finished sequences immediately so the next `step()` can
    /// reuse it. Blocks the prefix cache still holds stay resident.
    fn reclaim_finished(&mut self) {
        let mut finished = Vec::new();
        self.active_batch.retain(|seq| {
            if seq.is_finished {
                finished.push((seq.id, seq.request_id, seq.generated_tokens.clone()));
                false
            } else {
                true
            }
        });

        for (seq_id, request_id, tokens) in finished {
            self.kv.release_sequence(seq_id);
            let text = self.tokenizer.decode(&tokens, true).unwrap_or_default();
            println!(
                "[Req {request_id} / Seq {seq_id}] Finished, {} tokens: {}",
                tokens.len(),
                text.trim()
            );
        }
    }

    /// Runs the engine until all queues are empty.
    pub fn run(&mut self) -> Result<()> {
        let start_time = Instant::now();

        while !self.waiting_queue.is_empty() || !self.active_batch.is_empty() {
            self.step()?;
        }

        let elapsed = start_time.elapsed();
        let prefix = self.kv.prefix_stats();
        let s = &self.stats;
        let recomputed = s.prompt_tokens_prefilled;
        println!(
            "\nEngine run completed in {elapsed:.2?} over {} steps.",
            s.steps
        );
        println!(
            "  prompt tokens        : {} ({recomputed} prefilled, {} served from cache)",
            s.prompt_tokens,
            s.prompt_tokens - recomputed
        );
        println!("  generated tokens     : {}", s.generated_tokens);
        println!(
            "  prefix cache         : {} hits / {} lookups ({:.1}% hit rate), {} entries",
            prefix.hits,
            prefix.hits + prefix.misses,
            prefix.hit_rate() * 100.0,
            prefix.inserts - prefix.evictions,
        );
        println!("  copy-on-write copies : {}", self.kv.cow_copies());
        println!(
            "  prefill {:.2?} / decode {:.2?} ({:.1} tok/s decode)",
            s.prefill_time,
            s.decode_time,
            s.generated_tokens as f64 / s.decode_time.as_secs_f64().max(1e-9),
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    let block_size = 16;
    let total_blocks = 8192 / block_size;

    // Updated to target our TinyLlama 1.1B weights
    let tokenizer_path = "models/tinyllama-1.1b/tokenizer.json";
    let model_path = "models/tinyllama-1.1b/model.safetensors";

    if !std::path::Path::new(tokenizer_path).exists() || !std::path::Path::new(model_path).exists()
    {
        println!("Waiting on the TinyLlama 1.1B weights and tokenizer to download...");
        return Ok(());
    }

    // 1. Memory Map the Weights to OS Virtual Memory
    println!("Opening model file...");
    let file = File::open(model_path).expect("Could not open model file. Did it download?");
    let mmap = unsafe { MmapOptions::new().map(&file).expect("Failed to mmap file") };

    // 2. Parse the Safetensors layout into our Rust structs
    let loader = ModelLoader::new(&mmap).expect("Failed to initialize safetensors loader");
    let config = LlamaConfig::default();

    // The weights variable now safely holds the zero-copy mappings, anchored to `mmap`
    let weights = loader
        .load_weights(&config)
        .expect("Failed to map model weights");
    println!(
        "Successfully mapped all {} layers into memory without copying!",
        config.num_hidden_layers
    );

    // 3. Initialize the Continuous Batching Engine
    let mut engine = Engine::new(tokenizer_path, weights, config, total_blocks, block_size)?;

    // Three requests sharing one system prompt: the first pays to prefill it,
    // the other two map the same KV blocks and prefill only their own suffix.
    const SYSTEM: &str = "You are a systems engineer who explains GPU inference clearly and concisely. Answer the question directly.";
    engine.add_request(
        &format!("{SYSTEM}\n\nQuestion: What problem does PagedAttention solve?"),
        24,
    );
    engine.add_request(
        &format!("{SYSTEM}\n\nQuestion: Why does continuous batching improve throughput?"),
        24,
    );
    // Four continuations of one prompt: prefilled once, forked three times.
    engine.add_request_n(
        &format!("{SYSTEM}\n\nQuestion: Describe grouped-query attention."),
        24,
        4,
    );

    engine.run()?;

    Ok(())
}
