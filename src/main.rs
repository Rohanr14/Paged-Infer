use anyhow::Result;
use memmap2::MmapOptions;
use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::{LlamaConfig, ModelLoader};
use std::collections::VecDeque;
use std::fs::File;
use std::time::Instant;
use tokenizers::Tokenizer;

// TinyLlama (Llama 2 architecture) special tokens
const BOS_TOKEN: u32 = 1; // <s>
const EOS_TOKEN: u32 = 2; // </s>

/// Represents an incoming user request.
pub struct Request {
    pub id: usize,
    pub prompt: String,
    pub max_tokens: usize,
}

/// Represents an active generation sequence within the engine.
pub struct Sequence {
    pub id: usize,
    pub token_ids: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub block_table: BlockTable,
    pub is_finished: bool,
}

pub struct Engine<'a> {
    allocator: BlockAllocator,
    tokenizer: Tokenizer,
    waiting_queue: VecDeque<Request>,
    active_batch: Vec<Sequence>,
    next_request_id: usize,
    weights: LlamaWeights<'a>,
    config: LlamaConfig,
    kv_cache: Vec<f32>, // <--- The Actual Physical Memory Pool
}

impl Engine {
    pub fn new(tokenizer_path: &str, total_blocks: usize, block_size: usize) -> Result<Self> {
        println!("Initializing Paged-Infer Engine...");
        let allocator = BlockAllocator::new(total_blocks, block_size);
        println!("Allocated {} physical blocks ({} tokens per block).", total_blocks, block_size);

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        println!("Tokenizer loaded successfully.");

                // 22 layers * 2 (K&V) * 4 KV heads * 64 head_dim * total_blocks * block_size
        let kv_cache_size = 22 * 2 * 4 * 64 * total_blocks * block_size;
        let kv_cache = vec![0.0; kv_cache_size];
        println!("Allocated {:.2} MB for the Physical KV Cache.", (kv_cache_size * 4) as f32 / 1_048_576.0);

        Ok(Self {
            allocator,
            tokenizer,
            waiting_queue: VecDeque::new(),
            active_batch: Vec::new(),
            next_request_id: 0,
        })
    }

    /// Submits a new prompt to the engine's queue.
    pub fn add_request(&mut self, prompt: &str, max_tokens: usize) {
        self.waiting_queue.push_back(Request {
            id: self.next_request_id,
            prompt: prompt.to_string(),
            max_tokens,
        });
        self.next_request_id += 1;
    }

    /// Executes a single generation step across the active batch and manages the queue.
    pub fn step(&mut self) -> Result<()> {
        // --- PHASE 1: SCHEDULING (Prefill) ---
        while let Some(req) = self.waiting_queue.front() {
            let encoding = self.tokenizer.encode(req.prompt.clone(), true)
                .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
            
            // Prepend the Begin-Of-Sequence token
            let mut input_ids = vec![BOS_TOKEN];
            input_ids.extend_from_slice(encoding.get_ids());

            let initial_blocks_needed = (input_ids.len() + self.allocator.block_size - 1) / self.allocator.block_size;

            // Check if we have enough physical memory to admit this sequence
            if self.allocator.available_blocks() >= initial_blocks_needed {
                let req = self.waiting_queue.pop_front().unwrap();
                let mut block_table = BlockTable::new();

                for _ in 0..initial_blocks_needed {
                    let phys_block = self.allocator.allocate().unwrap();
                    block_table.append_block(phys_block);
                }

                println!("[Seq {}] Scheduled. Prompt len: {} tokens", req.id, input_ids.len());

                self.active_batch.push(Sequence {
                    id: req.id,
                    token_ids: input_ids,
                    generated_tokens: Vec::new(),
                    max_tokens: req.max_tokens,
                    block_table,
                    is_finished: false,
                });
            } else {
                // Not enough memory to admit new requests right now; wait for active ones to finish
                break;
            }
        }

        // --- PHASE 2: FORWARD PASS (Decode) ---
        for seq in self.active_batch.iter_mut() {
            if seq.is_finished {
                continue;
            }

            // Grab the last generated token (or the BOS token if starting)
            let current_token = *seq.token_ids.last().unwrap();
            let current_pos = seq.token_ids.len() - 1;

            // --- THE REAL FORWARD PASS ---
            let _logits = self.weights.forward(
                current_token,
                current_pos,
                &self.config,
                &seq.block_table,
                &mut self.kv_cache,
                self.allocator.block_size
            );

            // Argmax Sampling (Find the index of the highest logit)
            // Uncomment this once the math ops in model.rs are active
            // let next_token = _logits.iter()
            //      .enumerate()
            //      .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            //      .map(|(index, _)| index as u32)
            //      .unwrap();

            // Simulating EOS for now until the math ops return real logits
            let next_token = if seq.generated_tokens.len() >= seq.max_tokens - 1 {
                EOS_TOKEN
            } else {
                1000 // Arbitrary token ID for simulation
            };

            seq.generated_tokens.push(next_token);
            seq.token_ids.push(next_token);

            if next_token == EOS_TOKEN {
                seq.is_finished = true;
                let _decoded = self.tokenizer.decode(&seq.generated_tokens, true).unwrap_or_default();
                println!("[Seq {}] Finished generating. Output len: {}", seq.id, seq.generated_tokens.len());
            } else {
                // Check if the sequence needs a new physical block
                if seq.token_ids.len() % self.allocator.block_size == 1 {
                    if let Some(phys_block) = self.allocator.allocate() {
                        seq.block_table.append_block(phys_block);
                    } else {
                        println!("[Seq {}] Out of KV Cache! Forcing early termination.", seq.id);
                        seq.is_finished = true;
                    }
                }
            }
        }

        // --- PHASE 3: CLEANUP ---
        // Free memory for finished sequences immediately so the next `step()` can use it.
        self.active_batch.retain(|seq| {
            if seq.is_finished {
                for block in seq.block_table.mapped_blocks() {
                    self.allocator.free(*block);
                }
                false // Remove from active batch
            } else {
                true // Keep in active batch
            }
        });

        Ok(())
    }

    /// Runs the engine until all queues are empty.
    pub fn run(&mut self) -> Result<()> {
        let start_time = Instant::now();
        let mut step_count = 0;

        while !self.waiting_queue.is_empty() || !self.active_batch.is_empty() {
            self.step()?;
            step_count += 1;
        }

        println!("Engine run completed in {:.2?} over {} steps.", start_time.elapsed(), step_count);
        Ok(())
    }
}

fn main() -> Result<()> {
    let block_size = 16;
    let total_blocks = 8192 / block_size; 
    
    // Updated to target our TinyLlama 1.1B weights
    let tokenizer_path = "models/tinyllama-1.1b/tokenizer.json";
    let model_path = "models/tinyllama-1.1b/model.safetensors";

    if !std::path::Path::new(tokenizer_path).exists() || !std::path::Path::new(model_path).exists() {
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
    let _weights = loader.load_weights(&config).expect("Failed to map model weights");
    println!("Successfully mapped all {} layers into memory without copying!", config.num_hidden_layers);

    // 3. Initialize the Continuous Batching Engine
    let mut engine = Engine::new(tokenizer_path, total_blocks, block_size)?;
    
    // Simulate multiple incoming requests to test continuous batching
    engine.add_request("The architecture of a modern LLM inference engine requires", 15);
    engine.add_request("Rust is an excellent language for systems programming because", 10);
    engine.add_request("PagedAttention solves memory fragmentation by", 20);

    engine.run()?;

    Ok(())
}