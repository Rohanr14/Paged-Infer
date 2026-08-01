//! Paged attention over a ragged batch.
//!
//! Attention is the one part of a decode step that cannot amortize across the
//! batch. Every other operand — every projection, every FFN matrix — is shared,
//! so batching reads it once and reuses each row. The KV cache is not shared:
//! each sequence has its own history, its own position, its own scattered block
//! table. Batching cannot make attention cheaper *per sequence*.
//!
//! What it can do is stop making it more expensive than it has to be. Two costs
//! in the obvious implementation are pure overhead:
//!
//! **Grouped-query replication.** Under GQA, `kv_group` query heads share one
//! key/value head. Parallelizing over `(sequence, query head)` gives each of
//! those `kv_group` heads its own task, and each one streams the *same* K and V
//! vectors out of the cache independently. On TinyLlama that is eight
//! concurrent readers of the same bytes, on eight different cores, each pulling
//! its own copy through its own private cache. The traffic is real: at batch 8
//! with a long context it is the largest single term in the step. Here a lane
//! owns a whole group, so K and V are read once and used for every query head
//! that needs them.
//!
//! **Per-token address translation.** A logical token's physical slot is
//! `block_table[t / block_size]` plus `t % block_size` — a division, a modulo
//! and a bounds-checked lookup, done once per token, twice per pass, per head,
//! per layer. But the mapping is constant *within* a block, so it belongs
//! outside the token loop. Here it is resolved once per block and the tokens
//! inside are walked by stride.
//!
//! Neither changes a single arithmetic operation. Scores are still computed in
//! increasing `t`, the softmax still runs over the whole window at once, and the
//! value accumulation still runs in increasing `t` — so the output is
//! bit-identical to the per-head version, not merely close. `tests/attention_tests.rs`
//! asserts exactly that across every legal lane width.

use rayon::prelude::*;

use crate::math::{axpy, dot, softmax_in_place};
use crate::memory::block_table::BlockTable;
use crate::memory::layout::KvLayout;

/// Tasks per thread the lane scheduler aims for.
///
/// Read once. `PAGED_INFER_ATTN_LANES_PER_THREAD` exists so the trade can be
/// swept on a machine rather than argued about; the default was chosen by
/// measuring, and `attention_benchmark` is what measures it.
fn lanes_per_thread() -> usize {
    static SLACK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SLACK.get_or_init(|| {
        std::env::var("PAGED_INFER_ATTN_LANES_PER_THREAD")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(2)
    })
}

/// One batch entry, as attention sees it: a KV history and a place in it.
#[derive(Clone, Copy)]
pub struct AttnEntry<'a> {
    pub block_table: &'a BlockTable,
    /// Where this entry's query sits. It attends to `start..=pos`.
    pub pos: usize,
    /// First attended position — `0` for full causal attention, higher when a
    /// sliding window is configured.
    pub start: usize,
}

impl<'a> AttnEntry<'a> {
    /// An entry attending to everything up to `pos`, or to the last
    /// `window` positions when one is set.
    pub fn new(block_table: &'a BlockTable, pos: usize, window: Option<usize>) -> Self {
        let start = (pos + 1).saturating_sub(window.unwrap_or(usize::MAX));
        Self {
            block_table,
            pos,
            start,
        }
    }

    /// Positions this entry attends to.
    #[inline]
    pub fn window_len(&self) -> usize {
        (self.pos + 1).saturating_sub(self.start)
    }
}

/// Shape and scheduling decisions for one forward pass.
#[derive(Clone, Copy)]
pub struct PagedAttention {
    pub layout: KvLayout,
    pub block_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Query heads per key/value head.
    pub kv_group: usize,
    /// Distance between one query head's score lane and the next. Must be at
    /// least the widest window in the batch, and identical for every lane so
    /// the buffer can be split into equal chunks.
    pub score_stride: usize,
    /// Query heads handled by a single task. Divides `kv_group`.
    pub heads_per_lane: usize,
}

impl PagedAttention {
    /// How many query heads one task should own.
    ///
    /// This is the whole trade in one number. A wider lane reads K and V fewer
    /// times — `kv_group` heads off one read instead of one each — but produces
    /// proportionally fewer tasks, and tasks are what keep cores busy. Reading
    /// the cache once is worthless if three of four cores are idle waiting.
    ///
    /// So: take the widest lane that still leaves at least one task per thread.
    /// Batch 8 on TinyLlama gives `8 * 4 = 32` groups, plenty for any laptop, so
    /// lanes are full width and each K vector is read once. Batch 1 on a
    /// multi-query model would give a single group, so lanes narrow until there
    /// is work to go around. `d == 1` is exactly the per-head schedule this
    /// replaces, which is why that degenerate case is always available.
    ///
    /// Only divisors of `kv_group` are considered: a lane must not straddle two
    /// key/value heads, or it would need two K streams and the saving vanishes.
    ///
    /// "Enough tasks" is more than one per thread. Exactly one per thread looks
    /// balanced on paper and is not: rayon can only steal work that is still
    /// queued, so a schedule with no spare tasks runs at the speed of its
    /// slowest core with the others idle. [`lanes_per_thread`] sets the slack.
    pub fn lane_width(kv_group: usize, num_kv_heads: usize, batch: usize) -> usize {
        let want = rayon::current_num_threads().max(1) * lanes_per_thread();
        (1..=kv_group.max(1))
            .rev()
            .find(|d| kv_group.is_multiple_of(*d) && batch * num_kv_heads * (kv_group / d) >= want)
            .unwrap_or(1)
    }

    /// Scores and value-weighted sums for every head of every entry.
    ///
    /// `q` and `out` are `[entry][head][head_dim]`; `scores` is
    /// `[entry][head][score_stride]`. `kv_cache` is read-only here — every K/V
    /// write for this step must already have happened, which is what makes the
    /// output slices disjoint and the whole thing safe to run in parallel with
    /// no synchronization.
    pub fn run(
        &self,
        out: &mut [f32],
        scores: &mut [f32],
        q: &[f32],
        kv_cache: &[f32],
        entries: &[AttnEntry<'_>],
        layer: usize,
    ) {
        let batch = entries.len();
        let (d, head_dim, num_heads) = (self.heads_per_lane, self.head_dim, self.num_heads);

        // A lane derives its key/value head from its first query head, so the
        // head axis must tile exactly. A checkpoint whose `num_attention_heads`
        // is not a multiple of `num_key_value_heads` is not a valid GQA config,
        // but nothing stops one being loaded from a hand-edited `config.json` —
        // and the failure mode without this is a key/value head index past the
        // end of the layout, which reads whatever the next head's memory holds.
        assert!(
            self.kv_group > 0 && num_heads == self.kv_group * self.layout.num_kv_heads,
            "num_attention_heads ({num_heads}) must be num_key_value_heads ({}) \
             times the group size ({})",
            self.layout.num_kv_heads,
            self.kv_group
        );
        assert!(
            d > 0 && self.kv_group.is_multiple_of(d),
            "lane must divide the group"
        );
        assert_eq!(out.len(), batch * num_heads * head_dim);
        assert_eq!(q.len(), batch * num_heads * head_dim);
        assert_eq!(scores.len(), batch * num_heads * self.score_stride);
        // A zero stride would reach rayon as `par_chunks_mut(0)` and panic
        // inside a worker with a message about chunk sizes, which says nothing
        // about the configuration that caused it.
        assert!(
            batch == 0 || self.score_stride > 0,
            "score_stride must be at least 1; a zero-width attention window \
             should be `None`, not `Some(0)`"
        );
        // Lanes are equal-width so the buffer splits evenly, which means head
        // `g`'s score slice is only its own if the stride covers the window.
        // Undersized, `[g*stride..][..used]` overlaps head `g+1`'s region and
        // the heads silently overwrite each other's scores — and only the last
        // head is far enough out to panic, so the corruption lands first and
        // the backtrace points at the wrong head.
        debug_assert!(
            entries.iter().all(|e| e.window_len() <= self.score_stride),
            "score_stride {} is narrower than the widest window in the batch",
            self.score_stride
        );

        let scale = 1.0 / (head_dim as f32).sqrt();
        let lanes_per_entry = num_heads / d;
        // Distance between one token's K (or V) and the next token's, within a
        // block, for a fixed head. The layout interleaves heads and K/V between
        // consecutive tokens, so this is a stride rather than a step of one.
        let token_stride = self.layout.num_kv_heads * 2 * head_dim;
        let (block_size, layout) = (self.block_size, &self.layout);
        let score_stride = self.score_stride;

        out.par_chunks_mut(d * head_dim)
            .zip(scores.par_chunks_mut(d * score_stride))
            .enumerate()
            .for_each(|(lane, (out_lane, score_lane))| {
                let entry = entries[lane / lanes_per_entry];
                // Every head in this lane shares one key/value head, because `d`
                // divides `kv_group` and the lane starts on a multiple of `d`.
                let h0 = (lane % lanes_per_entry) * d;
                let kv_h = h0 / self.kv_group;
                // The chunk index doubles as the query offset: `hidden` is
                // `num_heads * head_dim`, so entry and head indices collapse.
                let q_lane = &q[lane * d * head_dim..(lane + 1) * d * head_dim];
                let used = entry.window_len();
                if used == 0 {
                    out_lane.fill(0.0);
                    return;
                }

                // Scores, one block-table lookup per block.
                for_each_block(entry, block_size, |si0, first_offset, take, block| {
                    match block {
                        Some(pb) => {
                            let base = layout.index(layer, pb, first_offset, kv_h, false);
                            for i in 0..take {
                                let k = &kv_cache[base + i * token_stride..][..head_dim];
                                for g in 0..d {
                                    let qh = &q_lane[g * head_dim..][..head_dim];
                                    score_lane[g * score_stride + si0 + i] = dot(qh, k) * scale;
                                }
                            }
                        }
                        // Unmapped: -inf, not 0.0. A zero would survive the
                        // softmax and take a full share of the weight.
                        None => {
                            for g in 0..d {
                                let lane = &mut score_lane[g * score_stride + si0..][..take];
                                lane.fill(f32::NEG_INFINITY);
                            }
                        }
                    }
                });

                for g in 0..d {
                    softmax_in_place(&mut score_lane[g * score_stride..][..used]);
                }

                out_lane.fill(0.0);
                for_each_block(entry, block_size, |si0, first_offset, take, block| {
                    let Some(pb) = block else { return };
                    let base = layout.index(layer, pb, first_offset, kv_h, true);
                    for i in 0..take {
                        let v = &kv_cache[base + i * token_stride..][..head_dim];
                        for g in 0..d {
                            let weight = score_lane[g * score_stride + si0 + i];
                            if weight == 0.0 {
                                continue;
                            }
                            axpy(&mut out_lane[g * head_dim..][..head_dim], weight, v);
                        }
                    }
                });
            });
    }
}

/// Walk `entry`'s attended window one physical block at a time.
///
/// Calls `f(score_offset, first_offset_in_block, count, physical_block)`. The
/// first and last blocks are usually partial: a window starts wherever the
/// sliding window puts it and ends at the entry's own position, neither of
/// which is block-aligned.
///
/// `block` is `None` for a logical block the sequence never mapped. That is a
/// property of the block, not of the tokens in it — `BlockTable` resolves
/// `t / block_size` and misses only when that index is past the end of the
/// mapping — so one lookup settles the whole run.
#[inline]
fn for_each_block(
    entry: AttnEntry<'_>,
    block_size: usize,
    mut f: impl FnMut(usize, usize, usize, Option<usize>),
) {
    let mut t = entry.start;
    while t <= entry.pos {
        let first_offset = t % block_size;
        let take = (block_size - first_offset).min(entry.pos + 1 - t);
        let block = entry
            .block_table
            .get_physical_location(t, block_size)
            .map(|(pb, _)| pb.index);
        f(t - entry.start, first_offset, take, block);
        t += take;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::allocator::PhysicalBlock;

    fn table(blocks: &[usize]) -> BlockTable {
        let mut t = BlockTable::new();
        for &b in blocks {
            t.append_block(PhysicalBlock { index: b });
        }
        t
    }

    /// Collect what `for_each_block` visits, as (score_offset, offset, count, block).
    fn visits(
        entry: AttnEntry<'_>,
        block_size: usize,
    ) -> Vec<(usize, usize, usize, Option<usize>)> {
        let mut v = Vec::new();
        for_each_block(entry, block_size, |a, b, c, d| v.push((a, b, c, d)));
        v
    }

    #[test]
    fn test_block_walk_covers_every_position_exactly_once() {
        let t = table(&[7, 3, 9, 1]);
        for block_size in [1usize, 2, 4, 8, 16] {
            for pos in 0..t.len() * block_size {
                for window in [None, Some(1), Some(3), Some(8), Some(1000)] {
                    let entry = AttnEntry::new(&t, pos, window);
                    let mut covered = Vec::new();
                    for (si0, off, take, block) in visits(entry, block_size) {
                        for i in 0..take {
                            covered.push(si0 + i);
                            // The physical slot the walk implies must equal what
                            // a per-token lookup would have produced.
                            let logical = entry.start + si0 + i;
                            let direct = t.get_physical_location(logical, block_size);
                            assert_eq!(block, direct.map(|(pb, _)| pb.index));
                            assert_eq!(off + i, direct.unwrap().1);
                        }
                    }
                    let expected: Vec<usize> = (0..entry.window_len()).collect();
                    assert_eq!(
                        covered, expected,
                        "block_size {block_size} pos {pos} window {window:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_block_walk_reports_unmapped_blocks_whole() {
        // Two blocks mapped, positions past them unmapped. Attention needs the
        // whole unmapped run in one call so it can mask it in one go.
        let t = table(&[5, 6]);
        let entry = AttnEntry::new(&t, 39, None);
        let v = visits(entry, 8);
        assert_eq!(
            v,
            vec![
                (0, 0, 8, Some(5)),
                (8, 0, 8, Some(6)),
                (16, 0, 8, None),
                (24, 0, 8, None),
                (32, 0, 8, None),
            ]
        );
    }

    #[test]
    fn test_an_empty_window_visits_nothing() {
        let t = table(&[0]);
        let entry = AttnEntry {
            block_table: &t,
            pos: 4,
            start: 5,
        };
        assert_eq!(entry.window_len(), 0);
        assert!(visits(entry, 8).is_empty());
    }

    #[test]
    fn test_lane_width_prefers_the_widest_lane_that_fills_the_threads() {
        let threads = rayon::current_num_threads().max(1) * lanes_per_thread();
        for kv_group in [1usize, 2, 4, 5, 8, 16] {
            for num_kv_heads in [1usize, 2, 4, 8] {
                for batch in [1usize, 2, 8, 32] {
                    let d = PagedAttention::lane_width(kv_group, num_kv_heads, batch);
                    assert!(d >= 1 && kv_group % d == 0, "d={d} must divide {kv_group}");
                    let lanes = batch * num_kv_heads * (kv_group / d);
                    // Either the schedule fills every thread, or it is already
                    // as fine-grained as the head axis allows.
                    assert!(
                        lanes >= threads || d == 1,
                        "kv_group {kv_group} kv_heads {num_kv_heads} batch {batch}: \
                         d={d} leaves {lanes} lanes for {threads} threads"
                    );
                    // And no wider divisor would also have filled them.
                    for wider in (d + 1)..=kv_group {
                        if kv_group % wider == 0 {
                            assert!(
                                batch * num_kv_heads * (kv_group / wider) < threads,
                                "d={d} chosen but {wider} would also have filled the threads"
                            );
                        }
                    }
                }
            }
        }
    }
}
