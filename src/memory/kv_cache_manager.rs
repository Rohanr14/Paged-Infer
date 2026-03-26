use std::collections::HashMap;

use super::allocator::{BlockAllocator, PhysicalBlock};

#[derive(Clone)]
struct SequenceAlloc {
    blocks: Vec<PhysicalBlock>,
    last_used_tick: u64,
}

pub struct KvCacheManager {
    allocator: BlockAllocator,
    sequences: HashMap<usize, SequenceAlloc>,
}

impl KvCacheManager {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(total_blocks, block_size),
            sequences: HashMap::new(),
        }
    }

    pub fn ensure_block_for_sequence(
        &mut self,
        seq_id: usize,
        now_tick: u64,
        allow_eviction: bool,
    ) -> bool {
        self.touch(seq_id, now_tick);

        if let Some(block) = self.allocator.allocate() {
            self.sequences
                .entry(seq_id)
                .or_insert(SequenceAlloc {
                    blocks: Vec::new(),
                    last_used_tick: now_tick,
                })
                .blocks
                .push(block);
            return true;
        }

        if !allow_eviction {
            return false;
        }

        if let Some(victim) = self.find_lru_victim(seq_id) {
            self.release_sequence(victim);
            if let Some(block) = self.allocator.allocate() {
                self.sequences
                    .entry(seq_id)
                    .or_insert(SequenceAlloc {
                        blocks: Vec::new(),
                        last_used_tick: now_tick,
                    })
                    .blocks
                    .push(block);
                return true;
            }
        }

        false
    }

    pub fn touch(&mut self, seq_id: usize, now_tick: u64) {
        if let Some(s) = self.sequences.get_mut(&seq_id) {
            s.last_used_tick = now_tick;
        }
    }

    pub fn release_sequence(&mut self, seq_id: usize) {
        if let Some(seq) = self.sequences.remove(&seq_id) {
            for b in seq.blocks {
                self.allocator.free(b);
            }
        }
    }

    pub fn active_sequences(&self) -> usize {
        self.sequences.len()
    }

    fn find_lru_victim(&self, exclude_seq: usize) -> Option<usize> {
        self.sequences
            .iter()
            .filter(|(sid, _)| **sid != exclude_seq)
            .min_by_key(|(_, s)| s.last_used_tick)
            .map(|(sid, _)| *sid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_eviction_allocates_when_full() {
        let mut mgr = KvCacheManager::new(2, 16);
        assert!(mgr.ensure_block_for_sequence(1, 1, false));
        assert!(mgr.ensure_block_for_sequence(2, 2, false));
        assert!(!mgr.ensure_block_for_sequence(3, 3, false));

        // With eviction, seq 1 is LRU and should be evicted.
        assert!(mgr.ensure_block_for_sequence(3, 4, true));
        assert_eq!(mgr.active_sequences(), 2);
    }
}
