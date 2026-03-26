use std::collections::VecDeque;

/// Represents a physical block in the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalBlock {
    pub index: usize,
}

pub struct BlockAllocator {
    pub block_size: usize,
    total_blocks: usize,
    free_blocks: VecDeque<PhysicalBlock>,
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
        }
    }

    /// Allocates a single physical block from the free pool.
    pub fn allocate(&mut self) -> Option<PhysicalBlock> {
        self.free_blocks.pop_front()
    }

    /// Returns a block to the free pool when a sequence finishes generation.
    pub fn free(&mut self, block: PhysicalBlock) {
        assert!(
            block.index < self.total_blocks,
            "Attempted to free an invalid physical block index."
        );
        self.free_blocks.push_back(block);
    }

    /// Returns the number of currently available blocks.
    pub fn available_blocks(&self) -> usize {
        self.free_blocks.len()
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
}