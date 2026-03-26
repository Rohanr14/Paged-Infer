use anyhow::Result;
use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;
use std::collections::VecDeque;
use std::time::Instant;
use tokenizers::Tokenizer;

const LLAMA3_BOS_TOKEN: u32 = 128000; // <|begin_of_text|>
const LLAMA3_EOS_TOKEN: u32 = 128009; // <|eot_id|>

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

pub struct Engine {
    allocator: BlockAllocator,
    tokenizer: Tokenizer,
    waiting_queue: VecDeque<Request>,
    active_batch: Vec<Sequence>,
    next_request_id: usize,
}

impl Engine {
    pub fn new(tokenizer_path: &str, total_blocks: usize, block_size: usize) -> Result<Self> {
        println!("Initializing Paged-Infer Engine...");
        let allocator = BlockAllocator::new(total_blocks, block_size);
        println!("Allocated {} physical blocks ({} tokens per block).", total_blocks, block_size);

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        println!("Tokenizer loaded successfully.");

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
        // Try to pull requests from the waiting queue if we have enough memory.
        while let Some(req) = self.waiting_queue.front() {
            let encoding = self.tokenizer.encode(req.prompt.clone(), true)
                .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
            
            // Prepend the Begin-Of-Sequence token
            let mut input_ids = vec![LLAMA3_BOS_TOKEN];
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
        // Here we would normally run `self.model.forward(&self.active_batch)`.
        for seq in self.active_batch.iter_mut() {
            if seq.is_finished {
                continue;
            }

            // SIMULATION: In reality, this comes from the model's logits.
            // We simulate predicting a token. We force EOS if it hits max_tokens.
            let next_token = if seq.generated_tokens.len() >= seq.max_tokens - 1 {
                LLAMA3_EOS_TOKEN
            } else {
                1000 // Arbitrary token ID for simulation
            };

            seq.generated_tokens.push(next_token);
            seq.token_ids.push(next_token);

            if next_token == LLAMA3_EOS_TOKEN {
                seq.is_finished = true;
                let decoded = self.tokenizer.decode(&seq.generated_tokens, true).unwrap_or_default();
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
    let tokenizer_path = "models/llama-3.2-1b/tokenizer.json";

    if !std::path::Path::new(tokenizer_path).exists() {
        println!("Waiting on the Llama 3.2 1B weights and tokenizer to download...");
        return Ok(());
    }

    let mut engine = Engine::new(tokenizer_path, total_blocks, block_size)?;
    
    // Simulate multiple incoming requests to test continuous batching
    engine.add_request("The architecture of a modern LLM inference engine requires", 15);
    engine.add_request("Rust is an excellent language for systems programming because", 10);
    engine.add_request("PagedAttention solves memory fragmentation by", 20);

    engine.run()?;

    Ok(())
}