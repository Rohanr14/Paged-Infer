use super::allocator::PhysicalBlock;

/// One logical block of a sequence, plus what the engine knows about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSlot {
    pub block: PhysicalBlock,
    /// Content hash of this block, chained through its predecessor. `Some` only
    /// once the block is full and its contents are therefore final.
    pub hash: Option<u64>,
    /// The block arrived from the prefix cache, so its KV state is already
    /// computed and prefill can skip the tokens it covers.
    pub cached: bool,
}

impl BlockSlot {
    pub fn fresh(block: PhysicalBlock) -> Self {
        Self {
            block,
            hash: None,
            cached: false,
        }
    }
}

/// Maps a sequence's logical blocks to the scattered physical blocks in the allocator.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    /// Each index represents a logical block in a sequence,
    /// containing the corresponding physical block.
    logical_to_physical: Vec<BlockSlot>,
}

impl BlockTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a new physical block to the sequence's mapping.
    pub fn append_block(&mut self, physical_block: PhysicalBlock) {
        self.logical_to_physical.push(BlockSlot::fresh(physical_block));
    }

    /// Appends a slot carrying prefix-cache provenance.
    pub fn append_slot(&mut self, slot: BlockSlot) {
        self.logical_to_physical.push(slot);
    }

    /// Given a logical token index, returns the physical block and the offset within that block.
    pub fn get_physical_location(
        &self,
        logical_token_idx: usize,
        block_size: usize,
    ) -> Option<(PhysicalBlock, usize)> {
        let logical_block_idx = logical_token_idx / block_size;
        let token_offset = logical_token_idx % block_size;

        self.logical_to_physical
            .get(logical_block_idx)
            .map(|slot| (slot.block, token_offset))
    }

    pub fn slots(&self) -> &[BlockSlot] {
        &self.logical_to_physical
    }

    pub fn slots_mut(&mut self) -> &mut [BlockSlot] {
        &mut self.logical_to_physical
    }

    /// Every physical block this sequence maps, in logical order.
    pub fn physical_blocks(&self) -> impl Iterator<Item = PhysicalBlock> + '_ {
        self.logical_to_physical.iter().map(|s| s.block)
    }

    pub fn len(&self) -> usize {
        self.logical_to_physical.len()
    }

    pub fn is_empty(&self) -> bool {
        self.logical_to_physical.is_empty()
    }

    /// Repoint a logical block at a different physical block. Used by
    /// copy-on-write once a shared block has been cloned.
    pub fn remap(&mut self, logical_block_idx: usize, block: PhysicalBlock) {
        let slot = &mut self.logical_to_physical[logical_block_idx];
        slot.block = block;
        // A cloned block is privately owned and no longer the cached original.
        slot.cached = false;
        slot.hash = None;
    }
}
