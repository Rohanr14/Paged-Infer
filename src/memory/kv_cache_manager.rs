//! Sequence-level ownership of KV blocks.
//!
//! Sits between the raw [`BlockAllocator`] and the scheduler, and owns the three
//! policies that decide who gets memory:
//!
//! * **Prefix reuse** — admitting a prompt maps already-computed blocks for its
//!   longest cached prefix instead of allocating and recomputing them.
//! * **Copy-on-write** — a block mapped by more than one sequence is cloned
//!   lazily, on the first write, so forked sequences pay only for what they
//!   actually diverge on.
//! * **Eviction** — under pressure, cold cache entries are dropped first and
//!   whole sequences are preempted only if that is not enough.

use std::collections::HashMap;

use super::allocator::{BlockAllocator, PhysicalBlock};
use super::block_table::{BlockSlot, BlockTable};
use super::layout::KvLayout;
use super::prefix_cache::{block_hash, PrefixCache, PrefixCacheStats, ROOT_HASH};

#[derive(Clone)]
struct SequenceAlloc {
    blocks: Vec<PhysicalBlock>,
    last_used_tick: u64,
}

/// Outcome of admitting a prompt.
#[derive(Debug, Clone)]
pub struct Admission {
    pub block_table: BlockTable,
    /// Leading prompt tokens whose KV came from the cache. Prefill starts here,
    /// so this is the work the request did not have to do.
    pub cached_tokens: usize,
    pub reused_blocks: usize,
    pub allocated_blocks: usize,
}

pub struct KvCacheManager {
    allocator: BlockAllocator,
    sequences: HashMap<usize, SequenceAlloc>,
    prefix_cache: PrefixCache,
    prefix_cache_enabled: bool,
    cow_copies: u64,
}

impl KvCacheManager {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(total_blocks, block_size),
            sequences: HashMap::new(),
            prefix_cache: PrefixCache::new(),
            prefix_cache_enabled: true,
            cow_copies: 0,
        }
    }

    pub fn with_prefix_cache(mut self, enabled: bool) -> Self {
        self.prefix_cache_enabled = enabled;
        self
    }

    pub fn block_size(&self) -> usize {
        self.allocator.block_size
    }

    pub fn available_blocks(&self) -> usize {
        self.allocator.available_blocks()
    }

    pub fn total_blocks(&self) -> usize {
        self.allocator.total_blocks()
    }

    pub fn allocator(&self) -> &BlockAllocator {
        &self.allocator
    }

    pub fn prefix_stats(&self) -> PrefixCacheStats {
        self.prefix_cache.stats()
    }

    pub fn cow_copies(&self) -> u64 {
        self.cow_copies
    }

    pub fn active_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Release every sequence and drop the prefix cache, returning the pool to
    /// its initial state. Intended for benchmarks measuring several
    /// configurations against one loaded model.
    ///
    /// Counters are zeroed along with the blocks: a hit rate that carried over
    /// from a previous configuration would describe neither of them.
    pub fn clear(&mut self) {
        let ids: Vec<usize> = self.sequences.keys().copied().collect();
        for id in ids {
            self.release_sequence(id);
        }
        self.prefix_cache.clear(&mut self.allocator);
        self.prefix_cache.reset_stats();
        self.cow_copies = 0;
        debug_assert_eq!(
            self.allocator.available_blocks(),
            self.allocator.total_blocks()
        );
    }

    // ── admission ────────────────────────────────────────────────────────────

    /// Map blocks for a prompt, reusing cached KV for its longest known prefix.
    ///
    /// Returns `None` if the prompt cannot be housed even after evicting cold
    /// cache entries; the caller should queue the request rather than admit it.
    pub fn admit(&mut self, seq_id: usize, tokens: &[u32], now_tick: u64) -> Option<Admission> {
        let block_size = self.allocator.block_size;
        let total_blocks = tokens.len().div_ceil(block_size).max(1);

        // Hash every *complete* block of the prompt. A trailing partial block
        // has no stable identity yet -- more tokens are still coming -- so it is
        // neither looked up nor published.
        let full_blocks = tokens.len() / block_size;
        let mut hashes = Vec::with_capacity(full_blocks);
        let mut parent = ROOT_HASH;
        for b in 0..full_blocks {
            parent = block_hash(parent, &tokens[b * block_size..(b + 1) * block_size]);
            hashes.push(parent);
        }

        // Walk the prefix, stopping at the first miss: past a divergence the
        // history differs, so no later block can legitimately match.
        let mut table = BlockTable::new();
        let mut reused = 0;
        if self.prefix_cache_enabled {
            for &hash in &hashes {
                match self.prefix_cache.lookup(hash) {
                    Some(block) => {
                        self.allocator.incref(block);
                        table.append_slot(BlockSlot {
                            block,
                            hash: Some(hash),
                            cached: true,
                        });
                        reused += 1;
                    }
                    None => break,
                }
            }
        }

        let to_allocate = total_blocks - reused;
        if self.allocator.available_blocks() < to_allocate
            && !self
                .prefix_cache
                .evict_until_available(&mut self.allocator, to_allocate)
        {
            // Roll back the references we took before giving up, so a rejected
            // admission leaves no trace.
            for slot in table.slots() {
                self.allocator.free(slot.block);
            }
            return None;
        }

        for i in reused..total_blocks {
            let block = self
                .allocator
                .allocate()
                .expect("capacity was just verified");
            table.append_slot(BlockSlot {
                block,
                hash: hashes.get(i).copied(),
                cached: false,
            });
        }

        let cached_tokens = reused * block_size;
        self.prefix_cache.record_tokens_saved(cached_tokens);

        self.sequences.insert(
            seq_id,
            SequenceAlloc {
                blocks: table.physical_blocks().collect(),
                last_used_tick: now_tick,
            },
        );

        Some(Admission {
            block_table: table,
            cached_tokens,
            reused_blocks: reused,
            allocated_blocks: to_allocate,
        })
    }

    /// Publish the prompt blocks this sequence computed, so later requests
    /// sharing the prefix can reuse them.
    ///
    /// Only blocks that were filled by the prompt are published: their contents
    /// are final and their hash is known. Blocks that fill during decode are
    /// not, since their contents depend on sampled tokens the hash chain does
    /// not cover.
    pub fn publish_prompt_blocks(&mut self, block_table: &BlockTable) -> usize {
        if !self.prefix_cache_enabled {
            return 0;
        }
        let mut published = 0;
        for slot in block_table.slots() {
            if let (Some(hash), false) = (slot.hash, slot.cached) {
                let before = self.prefix_cache.len();
                self.prefix_cache
                    .insert(hash, slot.block, &mut self.allocator);
                published += self.prefix_cache.len() - before;
            }
        }
        published
    }

    // ── growth ───────────────────────────────────────────────────────────────

    /// Add one block to a sequence that has run past the end of its mapping.
    pub fn append_block(
        &mut self,
        seq_id: usize,
        block_table: &mut BlockTable,
        now_tick: u64,
    ) -> bool {
        self.touch(seq_id, now_tick);
        if self.allocator.available_blocks() == 0
            && !self
                .prefix_cache
                .evict_until_available(&mut self.allocator, 1)
        {
            return false;
        }
        match self.allocator.allocate() {
            Some(block) => {
                block_table.append_block(block);
                self.sequences
                    .entry(seq_id)
                    .or_insert(SequenceAlloc {
                        blocks: Vec::new(),
                        last_used_tick: now_tick,
                    })
                    .blocks
                    .push(block);
                true
            }
            None => false,
        }
    }

    // ── copy-on-write ────────────────────────────────────────────────────────

    /// Fork a sequence: the child maps every one of the parent's blocks instead
    /// of copying them.
    ///
    /// This is how n samples from one prompt cost one prompt's worth of KV. No
    /// data moves here — divergence is handled later, and only for the one block
    /// both sequences actually write to.
    pub fn fork(
        &mut self,
        parent_table: &BlockTable,
        child_seq: usize,
        now_tick: u64,
    ) -> BlockTable {
        let mut child = BlockTable::new();
        for slot in parent_table.slots() {
            self.allocator.incref(slot.block);
            child.append_slot(*slot);
        }
        self.sequences.insert(
            child_seq,
            SequenceAlloc {
                blocks: child.physical_blocks().collect(),
                last_used_tick: now_tick,
            },
        );
        child
    }

    /// Make the block holding `token_pos` privately writable.
    ///
    /// If another sequence (or the prefix cache) maps the same block, clone it
    /// and repoint this sequence's table at the copy. Returns `true` if a copy
    /// was made. Copying is why this is lazy: a fork that stops immediately
    /// never pays it.
    pub fn ensure_writable(
        &mut self,
        seq_id: usize,
        block_table: &mut BlockTable,
        token_pos: usize,
        kv_cache: &mut [f32],
        layout: &KvLayout,
    ) -> bool {
        let logical = token_pos / self.allocator.block_size;
        let Some(slot) = block_table.slots().get(logical).copied() else {
            return false;
        };
        if !self.allocator.is_shared(slot.block) {
            return false;
        }
        if self.allocator.available_blocks() == 0
            && !self
                .prefix_cache
                .evict_until_available(&mut self.allocator, 1)
        {
            // Nothing to clone into. The caller has to preempt something.
            return false;
        }
        let Some(fresh) = self.allocator.allocate() else {
            return false;
        };

        layout.copy_block(kv_cache, slot.block, fresh);
        block_table.remap(logical, fresh);
        self.allocator.free(slot.block);

        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            if let Some(entry) = seq.blocks.iter_mut().find(|b| **b == slot.block) {
                *entry = fresh;
            }
        }
        self.cow_copies += 1;
        true
    }

    // ── lifecycle ────────────────────────────────────────────────────────────

    pub fn touch(&mut self, seq_id: usize, now_tick: u64) {
        if let Some(s) = self.sequences.get_mut(&seq_id) {
            s.last_used_tick = now_tick;
        }
    }

    /// Drop a sequence's references. Blocks the prefix cache still holds stay
    /// resident so a later request can reuse them.
    pub fn release_sequence(&mut self, seq_id: usize) {
        if let Some(seq) = self.sequences.remove(&seq_id) {
            for b in seq.blocks {
                self.allocator.free(b);
            }
        }
    }

    /// Legacy single-block path used by the eviction benchmark: take one block
    /// for a sequence, preempting the least recently used other sequence if the
    /// pool is dry.
    pub fn ensure_block_for_sequence(
        &mut self,
        seq_id: usize,
        now_tick: u64,
        allow_eviction: bool,
    ) -> bool {
        self.touch(seq_id, now_tick);

        if let Some(block) = self.allocator.allocate() {
            self.record(seq_id, block, now_tick);
            return true;
        }

        // Cold cache entries are cheaper to give up than a running sequence.
        if self.prefix_cache.evict_lru(&mut self.allocator) {
            if let Some(block) = self.allocator.allocate() {
                self.record(seq_id, block, now_tick);
                return true;
            }
        }

        if !allow_eviction {
            return false;
        }

        if let Some(victim) = self.find_lru_victim(seq_id) {
            self.release_sequence(victim);
            if let Some(block) = self.allocator.allocate() {
                self.record(seq_id, block, now_tick);
                return true;
            }
        }

        false
    }

    fn record(&mut self, seq_id: usize, block: PhysicalBlock, now_tick: u64) {
        self.sequences
            .entry(seq_id)
            .or_insert(SequenceAlloc {
                blocks: Vec::new(),
                last_used_tick: now_tick,
            })
            .blocks
            .push(block);
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

    const BS: usize = 4;

    fn prompt(prefix: &[u32], suffix: &[u32]) -> Vec<u32> {
        let mut v = prefix.to_vec();
        v.extend_from_slice(suffix);
        v
    }

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

    #[test]
    fn test_second_request_reuses_the_shared_prefix() {
        let mut mgr = KvCacheManager::new(32, BS);
        let shared: Vec<u32> = (0..12).collect();

        let a = mgr.admit(1, &prompt(&shared, &[90, 91]), 1).unwrap();
        assert_eq!(a.reused_blocks, 0, "cold cache cannot hit");
        assert_eq!(a.cached_tokens, 0);
        mgr.publish_prompt_blocks(&a.block_table);

        let b = mgr.admit(2, &prompt(&shared, &[80, 81]), 2).unwrap();
        assert_eq!(b.reused_blocks, 3, "all three full blocks are shared");
        assert_eq!(b.cached_tokens, 12);
        assert_eq!(b.allocated_blocks, 1, "only the divergent tail is new");

        // The two sequences map the same physical blocks for the shared prefix.
        for i in 0..3 {
            assert_eq!(
                a.block_table.slots()[i].block,
                b.block_table.slots()[i].block
            );
        }
        assert_eq!(mgr.prefix_stats().tokens_saved, 12);
    }

    #[test]
    fn test_reuse_stops_at_the_first_divergent_block() {
        let mut mgr = KvCacheManager::new(32, BS);
        let a = mgr
            .admit(1, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 1)
            .unwrap();
        mgr.publish_prompt_blocks(&a.block_table);

        // Same first block, different second block, then identical third.
        let b = mgr
            .admit(2, &[0, 1, 2, 3, 40, 41, 42, 43, 8, 9, 10, 11], 2)
            .unwrap();
        assert_eq!(
            b.reused_blocks, 1,
            "the third block has a different history and must not be reused"
        );
        assert_ne!(
            a.block_table.slots()[2].block,
            b.block_table.slots()[2].block
        );
    }

    #[test]
    fn test_disabling_the_cache_removes_all_reuse() {
        let mut mgr = KvCacheManager::new(32, BS).with_prefix_cache(false);
        let tokens: Vec<u32> = (0..12).collect();
        let a = mgr.admit(1, &tokens, 1).unwrap();
        assert_eq!(mgr.publish_prompt_blocks(&a.block_table), 0);
        let b = mgr.admit(2, &tokens, 2).unwrap();
        assert_eq!(b.reused_blocks, 0);
        assert_eq!(b.allocated_blocks, 3);
    }

    #[test]
    fn test_cached_blocks_survive_their_producer() {
        let mut mgr = KvCacheManager::new(8, BS);
        let tokens: Vec<u32> = (0..8).collect();
        let a = mgr.admit(1, &tokens, 1).unwrap();
        mgr.publish_prompt_blocks(&a.block_table);
        mgr.release_sequence(1);

        assert_eq!(mgr.available_blocks(), 6, "cache keeps its two blocks");
        let b = mgr.admit(2, &tokens, 2).unwrap();
        assert_eq!(b.reused_blocks, 2);
    }

    #[test]
    fn test_admission_under_pressure_evicts_cache_before_failing() {
        let mut mgr = KvCacheManager::new(4, BS);
        // Fill the cache and let its producer go.
        let a = mgr.admit(1, &[0, 1, 2, 3, 4, 5, 6, 7], 1).unwrap();
        mgr.publish_prompt_blocks(&a.block_table);
        mgr.release_sequence(1);
        assert_eq!(mgr.available_blocks(), 2);

        // Needs 4 fresh blocks; the two cached ones must be given up.
        let b = mgr.admit(
            2,
            &[
                90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
            ],
            2,
        );
        assert!(b.is_some(), "should reclaim cache rather than reject");
        assert!(mgr.prefix_stats().evictions >= 2);
    }

    #[test]
    fn test_rejected_admission_leaves_no_references_behind() {
        let mut mgr = KvCacheManager::new(4, BS);
        let a = mgr.admit(1, &[0, 1, 2, 3, 4, 5, 6, 7], 1).unwrap();
        mgr.publish_prompt_blocks(&a.block_table);
        // seq 1 stays alive, so its blocks cannot be reclaimed.
        let before = mgr.available_blocks();

        // Shares the prefix but needs far more than remains.
        let long: Vec<u32> = (0..8).chain(200..240).collect();
        assert!(mgr.admit(2, &long, 2).is_none());
        assert_eq!(
            mgr.available_blocks(),
            before,
            "a rejected admission must not strand blocks"
        );
        // The prefix blocks are still usable by someone who fits.
        let c = mgr.admit(3, &[0, 1, 2, 3, 4, 5, 6, 7], 3).unwrap();
        assert_eq!(c.reused_blocks, 2);
    }

    #[test]
    fn test_fork_shares_blocks_and_copies_on_write() {
        let layout = KvLayout::new(1, 8, BS, 1, 2);
        let mut cache = vec![0.0_f32; layout.total_floats()];
        let mut mgr = KvCacheManager::new(8, BS);

        let parent = mgr.admit(1, &[1, 2, 3, 4, 5, 6], 1).unwrap();
        let parent_table = parent.block_table;
        // Two blocks: one full, one holding two tokens.
        assert_eq!(parent_table.len(), 2);

        // Stamp recognisable state into the partial block.
        let idx = layout.index(0, parent_table.slots()[1].block.index, 0, 0, false);
        cache[idx] = 42.0;

        let before_free = mgr.available_blocks();
        let mut child_table = mgr.fork(&parent_table, 2, 2);
        assert_eq!(
            mgr.available_blocks(),
            before_free,
            "forking must not allocate"
        );
        assert_eq!(child_table.slots()[1].block, parent_table.slots()[1].block);

        // The child writes at position 6, inside the shared partial block.
        let copied = mgr.ensure_writable(2, &mut child_table, 6, &mut cache, &layout);
        assert!(copied, "a shared block must be cloned before writing");
        assert_ne!(child_table.slots()[1].block, parent_table.slots()[1].block);
        assert_eq!(mgr.cow_copies(), 1);

        // The clone carries the parent's state forward...
        let child_idx = layout.index(0, child_table.slots()[1].block.index, 0, 0, false);
        assert_eq!(cache[child_idx], 42.0);
        // ...and writing through it leaves the parent untouched.
        cache[layout.index(0, child_table.slots()[1].block.index, 2, 0, false)] = 7.0;
        assert_eq!(
            cache[layout.index(0, parent_table.slots()[1].block.index, 2, 0, false)],
            0.0
        );

        // The full block is still shared; nobody writes to it.
        assert_eq!(child_table.slots()[0].block, parent_table.slots()[0].block);

        // A second call is a no-op: the block is private now.
        assert!(!mgr.ensure_writable(2, &mut child_table, 6, &mut cache, &layout));
        assert_eq!(mgr.cow_copies(), 1);
    }

    #[test]
    fn test_unshared_block_is_never_copied() {
        let layout = KvLayout::new(1, 8, BS, 1, 2);
        let mut cache = vec![0.0_f32; layout.total_floats()];
        let mut mgr = KvCacheManager::new(8, BS);
        let admission = mgr.admit(1, &[1, 2, 3], 1).unwrap();
        let mut table = admission.block_table;
        assert!(!mgr.ensure_writable(1, &mut table, 2, &mut cache, &layout));
        assert_eq!(mgr.cow_copies(), 0);
    }

    #[test]
    fn test_release_after_fork_frees_only_the_last_holder() {
        let mut mgr = KvCacheManager::new(8, BS);
        let parent = mgr.admit(1, &[1, 2, 3, 4], 1).unwrap();
        let free_after_admit = mgr.available_blocks();
        let _child = mgr.fork(&parent.block_table, 2, 2);

        mgr.release_sequence(1);
        assert_eq!(
            mgr.available_blocks(),
            free_after_admit,
            "child still maps the blocks"
        );
        mgr.release_sequence(2);
        assert_eq!(mgr.available_blocks(), mgr.total_blocks());
    }

    #[test]
    fn test_churn_returns_every_block() {
        let mut mgr = KvCacheManager::new(64, BS);
        for round in 0..500_usize {
            let shared: Vec<u32> = (0..8).collect();
            let unique: Vec<u32> = (0..4).map(|i| (round * 10 + i) as u32).collect();
            let admission = mgr.admit(round, &prompt(&shared, &unique), round as u64);
            let admission = admission.expect("cache eviction should keep admissions feasible");
            mgr.publish_prompt_blocks(&admission.block_table);
            mgr.release_sequence(round);
        }
        assert_eq!(mgr.active_sequences(), 0);
        // Everything still held is held by the prefix cache, and clearing it
        // must return the pool to full.
        mgr.prefix_cache.clear(&mut mgr.allocator);
        assert_eq!(mgr.available_blocks(), 64, "blocks leaked across churn");
    }
}
