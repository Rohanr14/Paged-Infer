use std::collections::VecDeque;

/// Represents a physical block in the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalBlock {
    pub index: usize,
}

/// Owner of the physical KV cache blocks.
///
/// Blocks are reference counted rather than uniquely owned. That is what lets
/// several sequences map the same physical block at once — the basis for both
/// prefix-cache reuse (many requests sharing one system prompt) and
/// copy-on-write forking (n samples from one prompt). A block returns to the
/// free pool only when the last reference drops.
pub struct BlockAllocator {
    pub block_size: usize,
    total_blocks: usize,
    free_blocks: VecDeque<PhysicalBlock>,
    ref_counts: Vec<u32>,
}

impl BlockAllocator {
    /// Initializes a new allocator with a pre-determined pool of blocks.
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        // Initialize the free pool with all available physical block indices
        let mut free_blocks = VecDeque::with_capacity(total_blocks);
        for i in 0..total_blocks {
            free_blocks.push_back(PhysicalBlock { index: i });
        }

        Self {
            block_size,
            total_blocks,
            free_blocks,
            ref_counts: vec![0; total_blocks],
        }
    }

    /// Allocates a single physical block from the free pool, with one reference.
    pub fn allocate(&mut self) -> Option<PhysicalBlock> {
        let block = self.free_blocks.pop_front()?;
        debug_assert_eq!(self.ref_counts[block.index], 0, "free block had references");
        self.ref_counts[block.index] = 1;
        Some(block)
    }

    /// Take an additional reference to an already-live block.
    pub fn incref(&mut self, block: PhysicalBlock) {
        self.check(block);
        assert!(
            self.ref_counts[block.index] > 0,
            "incref on a free block (index {})",
            block.index
        );
        self.ref_counts[block.index] += 1;
    }

    /// Drop one reference. The block returns to the free pool at zero.
    ///
    /// Returns `true` if this call was the last reference.
    pub fn free(&mut self, block: PhysicalBlock) -> bool {
        self.check(block);
        let rc = &mut self.ref_counts[block.index];
        assert!(*rc > 0, "double free of block {}", block.index);
        *rc -= 1;
        if *rc == 0 {
            self.free_blocks.push_back(block);
            true
        } else {
            false
        }
    }

    /// How many references a block currently has (0 means free).
    pub fn ref_count(&self, block: PhysicalBlock) -> u32 {
        self.check(block);
        self.ref_counts[block.index]
    }

    /// True when more than one owner maps this block, so writing to it in place
    /// would corrupt someone else's KV state.
    pub fn is_shared(&self, block: PhysicalBlock) -> bool {
        self.ref_count(block) > 1
    }

    /// Returns the number of currently available blocks.
    pub fn available_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub fn allocated_blocks(&self) -> usize {
        self.total_blocks - self.free_blocks.len()
    }

    #[inline]
    fn check(&self, block: PhysicalBlock) {
        assert!(
            block.index < self.total_blocks,
            "physical block index {} out of range (total {})",
            block.index,
            self.total_blocks
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_cycle() {
        let mut allocator = BlockAllocator::new(100, 16);
        assert_eq!(allocator.available_blocks(), 100);

        let block = allocator.allocate().expect("Should allocate block");
        assert_eq!(allocator.available_blocks(), 99);

        allocator.free(block);
        assert_eq!(allocator.available_blocks(), 100);
    }

    #[test]
    fn test_shared_block_survives_until_last_reference_drops() {
        let mut allocator = BlockAllocator::new(4, 16);
        let block = allocator.allocate().unwrap();
        allocator.incref(block);
        allocator.incref(block);
        assert_eq!(allocator.ref_count(block), 3);
        assert!(allocator.is_shared(block));

        assert!(!allocator.free(block));
        assert_eq!(allocator.available_blocks(), 3);
        assert!(!allocator.free(block));
        assert_eq!(allocator.available_blocks(), 3);

        assert!(allocator.free(block), "last reference should reclaim");
        assert_eq!(allocator.available_blocks(), 4);
        assert_eq!(allocator.ref_count(block), 0);
        assert!(!allocator.is_shared(block));
    }

    #[test]
    fn test_exhaustion_and_recovery() {
        let mut allocator = BlockAllocator::new(3, 16);
        let blocks: Vec<_> = (0..3).map(|_| allocator.allocate().unwrap()).collect();
        assert!(allocator.allocate().is_none(), "pool should be exhausted");
        assert_eq!(allocator.allocated_blocks(), 3);

        allocator.free(blocks[1]);
        assert_eq!(allocator.allocate(), Some(blocks[1]));
    }

    #[test]
    fn test_many_cycles_do_not_leak() {
        // The Phase 2 deliverable: thousands of allocate/free cycles must return
        // the pool to exactly its starting state.
        let mut allocator = BlockAllocator::new(64, 16);
        for round in 0..5_000 {
            let take = (round % 64) + 1;
            let mut held = Vec::with_capacity(take);
            for _ in 0..take {
                held.push(allocator.allocate().expect("pool should not run dry"));
            }
            // Share every other block, so refcounts have to unwind correctly.
            for (i, b) in held.iter().enumerate() {
                if i % 2 == 0 {
                    allocator.incref(*b);
                }
            }
            for (i, b) in held.iter().enumerate() {
                if i % 2 == 0 {
                    allocator.free(*b);
                }
                allocator.free(*b);
            }
            assert_eq!(allocator.available_blocks(), 64, "leak after round {round}");
        }
        for i in 0..64 {
            assert_eq!(allocator.ref_count(PhysicalBlock { index: i }), 0);
        }
    }

    #[test]
    #[should_panic(expected = "double free")]
    fn test_double_free_is_caught() {
        let mut allocator = BlockAllocator::new(2, 16);
        let block = allocator.allocate().unwrap();
        allocator.free(block);
        allocator.free(block);
    }

    #[test]
    #[should_panic(expected = "incref on a free block")]
    fn test_incref_after_free_is_caught() {
        let mut allocator = BlockAllocator::new(2, 16);
        let block = allocator.allocate().unwrap();
        allocator.free(block);
        allocator.incref(block);
    }
}
