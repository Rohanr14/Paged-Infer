use super::allocator::PhysicalBlock;

/// Maps a sequence's logical blocks to the scattered physical blocks in the allocator.
pub struct BlockTable {
    /// Each index represents a logical block in a sequence, 
    /// containing the corresponding physical block.
    logical_to_physical: Vec<PhysicalBlock>,
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            logical_to_physical: Vec::new(),
        }
    }

    /// Appends a new physical block to the sequence's mapping.
    pub fn append_block(&mut self, physical_block: PhysicalBlock) {
        self.logical_to_physical.push(physical_block);
    }

    /// Given a logical token index, returns the physical block and the offset within that block.
    pub fn get_physical_location(&self, logical_token_idx: usize, block_size: usize) -> Option<(PhysicalBlock, usize)> {
        let logical_block_idx = logical_token_idx / block_size;
        let token_offset = logical_token_idx % block_size;

        if logical_block_idx < self.logical_to_physical.len() {
            Some((self.logical_to_physical[logical_block_idx], token_offset))
        } else {
            None
        }
    }
    
    pub fn mapped_blocks(&self) -> &[PhysicalBlock] {
        &self.logical_to_physical
    }
}