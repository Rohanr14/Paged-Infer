//! Content-addressed reuse of already-computed KV blocks.
//!
//! Serving traffic is enormously repetitive: every request in a deployment
//! tends to carry the same system prompt, the same few-shot preamble, or the
//! same document. Recomputing that prefix per request is pure waste — the KV
//! state for a given token sequence is a deterministic function of that
//! sequence, so it can be computed once and shared.
//!
//! A block is keyed by the hash of *every* token that precedes and includes it,
//! chained through the previous block's hash. That chaining is what makes reuse
//! sound: two requests may share the same 16 tokens at some offset while having
//! completely different histories, and their KV state for those tokens would
//! then differ. Keying on the whole prefix means a hit implies identical
//! history, which is exactly the condition under which the cached KV is valid.
//!
//! Because keys are prefixes, a lookup walks blocks in order and stops at the
//! first miss — everything after a divergence has a different history and
//! cannot match.

use std::collections::HashMap;

use super::allocator::{BlockAllocator, PhysicalBlock};

/// FNV-1a over the previous block's hash followed by this block's tokens.
///
/// Chaining through `parent` makes a block's key depend on its entire prefix.
/// Pass `ROOT_HASH` for the first block of a sequence.
pub fn block_hash(parent: u64, tokens: &[u32]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut h = parent ^ 0xcbf2_9ce4_8422_2325;
    for &t in tokens {
        for byte in t.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(PRIME);
        }
    }
    // Fold in the length so a short block can never alias a longer one.
    h ^= tokens.len() as u64;
    h.wrapping_mul(PRIME)
}

/// Seed for the first block of a sequence.
pub const ROOT_HASH: u64 = 0;

#[derive(Debug, Clone, Copy)]
struct Entry {
    block: PhysicalBlock,
    last_used: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    /// Block lookups that found a usable cached block.
    pub hits: u64,
    /// Block lookups that missed.
    pub misses: u64,
    /// Blocks published into the cache.
    pub inserts: u64,
    /// Blocks dropped from the cache to reclaim memory.
    pub evictions: u64,
    /// Prompt tokens whose KV never had to be recomputed.
    pub tokens_saved: u64,
}

impl PrefixCacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// A map from prefix hash to a physical block holding that prefix's KV state.
///
/// The cache holds one reference on every block it stores, so a cached block
/// survives the sequence that produced it. It never frees blocks on its own —
/// reclamation goes through [`PrefixCache::evict_lru`], which the owner calls
/// under memory pressure.
#[derive(Debug, Default)]
pub struct PrefixCache {
    entries: HashMap<u64, Entry>,
    tick: u64,
    stats: PrefixCacheStats,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a block by prefix hash, recording a hit or a miss.
    ///
    /// The caller must `incref` the returned block before mapping it into a
    /// sequence — the reference the cache holds is its own.
    pub fn lookup(&mut self, hash: u64) -> Option<PhysicalBlock> {
        self.tick += 1;
        let tick = self.tick;
        match self.entries.get_mut(&hash) {
            Some(entry) => {
                entry.last_used = tick;
                self.stats.hits += 1;
                Some(entry.block)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Publish a fully-computed block. Takes a reference on it.
    ///
    /// A second insert for the same hash is a no-op: the existing entry is just
    /// as valid, and keeping it avoids churning references for no benefit.
    pub fn insert(&mut self, hash: u64, block: PhysicalBlock, allocator: &mut BlockAllocator) {
        if self.entries.contains_key(&hash) {
            return;
        }
        self.tick += 1;
        allocator.incref(block);
        self.entries.insert(
            hash,
            Entry {
                block,
                last_used: self.tick,
            },
        );
        self.stats.inserts += 1;
    }

    /// Drop the least recently used entry whose block nothing else is using,
    /// returning it to the free pool. Returns `true` if a block was reclaimed.
    ///
    /// Entries whose block is still mapped by a live sequence are skipped
    /// rather than evicted: dropping the cache's reference would not free
    /// anything, and would throw away a hit the moment the sequence finishes.
    pub fn evict_lru(&mut self, allocator: &mut BlockAllocator) -> bool {
        let victim = self
            .entries
            .iter()
            .filter(|(_, e)| allocator.ref_count(e.block) == 1)
            .min_by_key(|(_, e)| e.last_used)
            .map(|(h, _)| *h);

        match victim {
            Some(hash) => {
                let entry = self.entries.remove(&hash).expect("victim just found");
                allocator.free(entry.block);
                self.stats.evictions += 1;
                true
            }
            None => false,
        }
    }

    /// Reclaim until `wanted` blocks are free, or until nothing more can go.
    pub fn evict_until_available(&mut self, allocator: &mut BlockAllocator, wanted: usize) -> bool {
        while allocator.available_blocks() < wanted {
            if !self.evict_lru(allocator) {
                return false;
            }
        }
        true
    }

    /// Drop every entry, releasing the cache's references.
    pub fn clear(&mut self, allocator: &mut BlockAllocator) {
        for (_, entry) in self.entries.drain() {
            allocator.free(entry.block);
            self.stats.evictions += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> PrefixCacheStats {
        self.stats
    }

    /// Zero the counters without touching the entries.
    ///
    /// Used when the owner returns the pool to its initial state: a reported
    /// hit rate should describe one workload, not every workload the process
    /// has ever run.
    pub fn reset_stats(&mut self) {
        self.stats = PrefixCacheStats::default();
    }

    pub(crate) fn record_tokens_saved(&mut self, tokens: usize) {
        self.stats.tokens_saved += tokens as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_depends_on_the_whole_prefix() {
        // The same 16 tokens reached through different histories must not
        // collide -- their KV state genuinely differs.
        let tokens = [7_u32; 16];
        let a = block_hash(block_hash(ROOT_HASH, &[1, 2, 3]), &tokens);
        let b = block_hash(block_hash(ROOT_HASH, &[1, 2, 4]), &tokens);
        assert_ne!(a, b, "chained hash ignored its parent");

        // Same history, same tokens -> same key.
        let c = block_hash(block_hash(ROOT_HASH, &[1, 2, 3]), &tokens);
        assert_eq!(a, c);
    }

    #[test]
    fn test_hash_is_order_and_length_sensitive() {
        assert_ne!(
            block_hash(ROOT_HASH, &[1, 2]),
            block_hash(ROOT_HASH, &[2, 1])
        );
        assert_ne!(
            block_hash(ROOT_HASH, &[1, 2]),
            block_hash(ROOT_HASH, &[1, 2, 0])
        );
    }

    #[test]
    fn test_lookup_hit_and_miss_accounting() {
        let mut allocator = BlockAllocator::new(4, 16);
        let mut cache = PrefixCache::new();
        let block = allocator.allocate().unwrap();

        assert!(cache.lookup(42).is_none());
        cache.insert(42, block, &mut allocator);
        assert_eq!(
            allocator.ref_count(block),
            2,
            "cache should hold a reference"
        );
        assert_eq!(cache.lookup(42), Some(block));

        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.inserts), (1, 1, 1));
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cached_block_outlives_its_producer() {
        let mut allocator = BlockAllocator::new(2, 16);
        let mut cache = PrefixCache::new();
        let block = allocator.allocate().unwrap();
        cache.insert(1, block, &mut allocator);

        // The producing sequence finishes and drops its reference.
        allocator.free(block);
        assert_eq!(allocator.ref_count(block), 1);
        assert_eq!(
            allocator.available_blocks(),
            1,
            "block must not be reclaimed"
        );
        assert_eq!(cache.lookup(1), Some(block));
    }

    #[test]
    fn test_evict_lru_skips_blocks_still_in_use() {
        let mut allocator = BlockAllocator::new(4, 16);
        let mut cache = PrefixCache::new();

        let in_use = allocator.allocate().unwrap();
        let idle = allocator.allocate().unwrap();
        cache.insert(1, in_use, &mut allocator);
        cache.insert(2, idle, &mut allocator);
        // Only `idle`'s producer has finished.
        allocator.free(idle);

        assert!(cache.evict_lru(&mut allocator));
        assert_eq!(cache.lookup(2), None, "idle entry should have been evicted");
        assert_eq!(cache.lookup(1), Some(in_use), "in-use entry must survive");

        // Nothing left that can be reclaimed.
        assert!(!cache.evict_lru(&mut allocator));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_evict_lru_picks_the_coldest_entry() {
        let mut allocator = BlockAllocator::new(8, 16);
        let mut cache = PrefixCache::new();
        let blocks: Vec<_> = (0..3).map(|_| allocator.allocate().unwrap()).collect();
        for (i, b) in blocks.iter().enumerate() {
            cache.insert(i as u64, *b, &mut allocator);
            allocator.free(*b);
        }
        // Touch 0 and 2, leaving 1 coldest.
        cache.lookup(0);
        cache.lookup(2);

        assert!(cache.evict_lru(&mut allocator));
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(1).is_none(), "coldest entry should be evicted");
    }

    #[test]
    fn test_clear_releases_every_reference() {
        let mut allocator = BlockAllocator::new(4, 16);
        let mut cache = PrefixCache::new();
        for i in 0..4 {
            let b = allocator.allocate().unwrap();
            cache.insert(i, b, &mut allocator);
            allocator.free(b);
        }
        assert_eq!(allocator.available_blocks(), 0);
        cache.clear(&mut allocator);
        assert_eq!(allocator.available_blocks(), 4);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_duplicate_insert_does_not_double_reference() {
        let mut allocator = BlockAllocator::new(2, 16);
        let mut cache = PrefixCache::new();
        let block = allocator.allocate().unwrap();
        cache.insert(9, block, &mut allocator);
        cache.insert(9, block, &mut allocator);
        assert_eq!(allocator.ref_count(block), 2);
        assert_eq!(cache.stats().inserts, 1);
    }
}
