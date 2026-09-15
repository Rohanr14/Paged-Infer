//! Decode-only shared-prefix scheduling with full-window normalization.
//!
//! Each lane owns a GQA head group from up to four sequences. The common K and V
//! vectors are loaded by SIMD multi-output primitives. Each query still has its
//! own scores and whole-window softmax, and values accumulate in token order.

use rayon::prelude::*;

use super::{for_each_block, AttnEntry, PagedAttention};
use crate::math::{axpy, dot, softmax_in_place};
use crate::simd::{dot_multi, weighted_sum_multi};

const TILE: usize = 4;

/// Reusable packed query, score and output storage. Contains no KV identities.
#[derive(Default)]
pub struct SharedPrefixScratch {
    work: Vec<f32>,
}

impl SharedPrefixScratch {
    /// Allocated capacity, including padding for an incomplete query tile.
    pub fn allocated_bytes(&self) -> usize {
        self.work.capacity() * std::mem::size_of::<f32>()
    }
}

/// One prefix common to a complete decode batch, borrowed for this forward pass.
///
/// Matching physical blocks must appear at the same logical positions. Every
/// current write block is excluded, and the shared interval is intersected with
/// all query windows. The table borrow prevents remapping while this plan lives;
/// callers must rebuild it after COW, cancellation, rollback or block recycling.
/// KV writes for a layer must finish before `run`, as with ordinary attention.
pub struct SharedPrefixPlan<'a> {
    entries: &'a [AttnEntry<'a>],
    block_size: usize,
    start: usize,
    end: usize,
}

impl<'a> SharedPrefixPlan<'a> {
    pub fn new(entries: &'a [AttnEntry<'a>], block_size: usize) -> Option<Self> {
        assert!(block_size > 0, "block_size must be positive");
        if entries.len() < 2 {
            return None;
        }
        let limit = entries
            .iter()
            .map(|e| (e.pos / block_size).min(e.block_table.len()))
            .min()
            .unwrap_or(0);
        let first = entries[0].block_table.slots();
        let blocks = (0..limit)
            .take_while(|&i| {
                entries[1..]
                    .iter()
                    .all(|e| e.block_table.slots()[i].block == first[i].block)
            })
            .count();
        let end = blocks * block_size;
        let start = entries.iter().map(|e| e.start).max().unwrap_or(0);
        (start < end).then_some(Self {
            entries,
            block_size,
            start,
            end,
        })
    }

    /// Positions in the common attended interval, once per query head/entry.
    pub fn prefix_tokens(&self) -> usize {
        self.end - self.start
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        attn: &PagedAttention,
        out: &mut [f32],
        q: &[f32],
        kv_cache: &[f32],
        layer: usize,
        scratch: &mut SharedPrefixScratch,
    ) {
        use crate::profiling::{Span, Stage};
        let setup_profile = Span::new(Stage::SharedSetup);
        let batch = self.entries.len();
        let (heads, dim, stride) = (attn.num_heads, attn.head_dim, attn.score_stride);
        assert_eq!(attn.block_size, self.block_size);
        assert!(dim > 0 && heads > 0 && attn.kv_group > 0);
        assert_eq!(heads, attn.kv_group * attn.layout.num_kv_heads);
        assert_eq!(dim, attn.layout.head_dim);
        assert_eq!(out.len(), batch * heads * dim);
        assert_eq!(q.len(), out.len());
        assert!(self.entries.iter().all(|e| e.window_len() <= stride));

        // Keep GQA heads together for private histories too. Splitting them
        // into unrelated tasks would undo the fallback kernel's cache locality.
        let tiles = batch.div_ceil(TILE);
        let d = PagedAttention::lane_width(attn.kv_group, attn.layout.num_kv_heads, tiles);
        let head_lanes = heads / d;
        let lane_len = TILE * d * (2 * dim + stride);
        let lane_count = tiles * head_lanes;
        let work_len = lane_count * lane_len;
        if scratch.work.len() < work_len {
            scratch.work.resize(work_len, 0.0);
        }
        let scale = 1.0 / (dim as f32).sqrt();
        let token_stride = attn.layout.num_kv_heads * 2 * dim;
        // Query reuse helps long common prefixes in the measured Llama shape,
        // but regresses shorter contexts and TinyLlama's wider GQA groups.
        // Retain the established score schedule outside that measured range.
        #[cfg(target_arch = "aarch64")]
        let pair_shared_keys = heads == 32
            && attn.layout.num_kv_heads == 8
            && dim == 64
            && self.prefix_tokens() >= 2048;
        drop(setup_profile);
        scratch.work[..work_len]
            .par_chunks_mut(lane_len)
            .enumerate()
            .for_each(|(lane, work)| {
                let mut profile = Span::new(Stage::SharedPack);
                let b0 = lane / head_lanes * TILE;
                let h0 = lane % head_lanes * d;
                let kv_h = h0 / attn.kv_group;
                let count = (batch - b0).min(TILE);
                let (queries, rest) = work.split_at_mut(TILE * d * dim);
                let (values, scores) = rest.split_at_mut(TILE * d * dim);
                queries.fill(0.0);
                values.fill(0.0);
                for row in 0..count {
                    for g in 0..d {
                        let offset = ((b0 + row) * heads + h0 + g) * dim;
                        queries[(g * TILE + row) * dim..][..dim]
                            .copy_from_slice(&q[offset..][..dim]);
                    }
                    profile.enter(Stage::PrivateScores);
                    self.private_scores(
                        attn,
                        scores,
                        queries,
                        kv_cache,
                        layer,
                        kv_h,
                        row,
                        d,
                        self.entries[b0 + row],
                        self.entries[b0 + row].start,
                        self.start,
                        scale,
                    );
                    profile.enter(Stage::SharedPack);
                }
                let shared = AttnEntry {
                    block_table: self.entries[0].block_table,
                    start: self.start,
                    pos: self.end - 1,
                };
                profile.enter(Stage::SharedScores);
                for_each_block(shared, self.block_size, |si, off, take, pb| {
                    let base = attn.layout.index(
                        layer,
                        pb.expect("shared mapped prefix"),
                        off,
                        kv_h,
                        false,
                    );
                    #[cfg(target_arch = "aarch64")]
                    let first_single_key = {
                        let mut i = 0;
                        if pair_shared_keys && count > 1 {
                            // Pair keys only inside this physical block. Each
                            // 2-query tile reuses its query loads across both
                            // keys while retaining independent dot reductions.
                            while i + 1 < take {
                                let keys =
                                    &kv_cache[base + i * token_stride..][..token_stride + dim];
                                for g in 0..d {
                                    for row0 in (0..count).step_by(2) {
                                        let query_pair =
                                            &queries[(g * TILE + row0) * dim..][..2 * dim];
                                        let dots = crate::simd::neon::dot_queries2_keys2(
                                            query_pair,
                                            dim,
                                            keys,
                                            token_stride,
                                            dim,
                                        );
                                        for (key_offset, key_dots) in dots.iter().enumerate() {
                                            for (query_offset, dot) in key_dots
                                                .iter()
                                                .take((count - row0).min(2))
                                                .enumerate()
                                            {
                                                let row = row0 + query_offset;
                                                let offset = self.start
                                                    - self.entries[b0 + row].start
                                                    + si
                                                    + i
                                                    + key_offset;
                                                scores[(g * TILE + row) * stride + offset] =
                                                    dot * scale;
                                            }
                                        }
                                    }
                                }
                                i += 2;
                            }
                        }
                        i
                    };
                    #[cfg(not(target_arch = "aarch64"))]
                    let first_single_key = 0;
                    for i in first_single_key..take {
                        let k = &kv_cache[base + i * token_stride..][..dim];
                        for g in 0..d {
                            let query_group = &queries[g * TILE * dim..][..TILE * dim];
                            let dots = match count {
                                1 => [dot(&query_group[..dim], k), 0.0, 0.0, 0.0],
                                2 => {
                                    let dots = dot_multi::<2>(k, query_group, dim);
                                    [dots[0], dots[1], 0.0, 0.0]
                                }
                                _ => dot_multi::<TILE>(k, query_group, dim),
                            };
                            for row in 0..count {
                                let offset = self.start - self.entries[b0 + row].start + si + i;
                                scores[(g * TILE + row) * stride + offset] = dots[row] * scale;
                            }
                        }
                    }
                });
                for row in 0..count {
                    let entry = self.entries[b0 + row];
                    profile.enter(Stage::PrivateScores);
                    self.private_scores(
                        attn,
                        scores,
                        queries,
                        kv_cache,
                        layer,
                        kv_h,
                        row,
                        d,
                        entry,
                        self.end,
                        entry.pos + 1,
                        scale,
                    );
                    profile.enter(Stage::Softmax);
                    for g in 0..d {
                        softmax_in_place(
                            &mut scores[(g * TILE + row) * stride..][..entry.window_len()],
                        );
                    }
                    profile.enter(Stage::PrivateValues);
                    self.private_values(
                        attn,
                        values,
                        scores,
                        kv_cache,
                        layer,
                        kv_h,
                        row,
                        d,
                        entry,
                        entry.start,
                        self.start,
                    );
                }
                profile.enter(Stage::SharedValues);
                for_each_block(shared, self.block_size, |si, off, take, pb| {
                    let base = attn.layout.index(
                        layer,
                        pb.expect("shared mapped prefix"),
                        off,
                        kv_h,
                        true,
                    );
                    let value_block = &kv_cache[base..][..(take - 1) * token_stride + dim];
                    for g in 0..d {
                        let weights = |row: usize| {
                            let offset = (g * TILE + row) * stride + self.start
                                - self.entries[b0 + row].start
                                + si;
                            &scores[offset..][..take]
                        };
                        let output = &mut values[g * TILE * dim..][..TILE * dim];
                        match count {
                            1 => weighted_sum_multi::<1>(
                                output,
                                dim,
                                [weights(0)],
                                value_block,
                                token_stride,
                                dim,
                            ),
                            2 => weighted_sum_multi::<2>(
                                output,
                                dim,
                                [weights(0), weights(1)],
                                value_block,
                                token_stride,
                                dim,
                            ),
                            3 => weighted_sum_multi::<3>(
                                output,
                                dim,
                                [weights(0), weights(1), weights(2)],
                                value_block,
                                token_stride,
                                dim,
                            ),
                            _ => weighted_sum_multi::<4>(
                                output,
                                dim,
                                [weights(0), weights(1), weights(2), weights(3)],
                                value_block,
                                token_stride,
                                dim,
                            ),
                        }
                    }
                });
                for row in 0..count {
                    let entry = self.entries[b0 + row];
                    profile.enter(Stage::PrivateValues);
                    self.private_values(
                        attn,
                        values,
                        scores,
                        kv_cache,
                        layer,
                        kv_h,
                        row,
                        d,
                        entry,
                        self.end,
                        entry.pos + 1,
                    );
                }
            });

        let _scatter_profile = Span::new(Stage::SharedScatter);
        // Packed lanes are disjoint during parallel computation. Scatter only
        // afterwards, avoiding aliased mutable slices or raw output pointers.
        for (lane, work) in scratch.work[..work_len].chunks_exact(lane_len).enumerate() {
            let b0 = lane / head_lanes * TILE;
            let h0 = lane % head_lanes * d;
            for row in 0..(batch - b0).min(TILE) {
                for g in 0..d {
                    let offset = ((b0 + row) * heads + h0 + g) * dim;
                    let packed = TILE * d * dim + (g * TILE + row) * dim;
                    out[offset..][..dim].copy_from_slice(&work[packed..][..dim]);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn private_scores(
        &self,
        attn: &PagedAttention,
        scores: &mut [f32],
        queries: &[f32],
        kv: &[f32],
        layer: usize,
        kv_h: usize,
        row: usize,
        d: usize,
        entry: AttnEntry<'_>,
        start: usize,
        end: usize,
        scale: f32,
    ) {
        if start >= end {
            return;
        }
        let dim = attn.head_dim;
        let token_stride = attn.layout.num_kv_heads * 2 * dim;
        for_each_block(
            AttnEntry {
                start,
                pos: end - 1,
                ..entry
            },
            self.block_size,
            |si, off, take, pb| {
                if let Some(pb) = pb {
                    let base = attn.layout.index(layer, pb, off, kv_h, false);
                    for i in 0..take {
                        let key = &kv[base + i * token_stride..][..dim];
                        for g in 0..d {
                            let query = &queries[(g * TILE + row) * dim..][..dim];
                            let offset =
                                (g * TILE + row) * attn.score_stride + start - entry.start + si + i;
                            scores[offset] = dot(query, key) * scale;
                        }
                    }
                } else {
                    for g in 0..d {
                        let offset =
                            (g * TILE + row) * attn.score_stride + start - entry.start + si;
                        scores[offset..][..take].fill(f32::NEG_INFINITY);
                    }
                }
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn private_values(
        &self,
        attn: &PagedAttention,
        values: &mut [f32],
        scores: &[f32],
        kv: &[f32],
        layer: usize,
        kv_h: usize,
        row: usize,
        d: usize,
        entry: AttnEntry<'_>,
        start: usize,
        end: usize,
    ) {
        if start >= end {
            return;
        }
        let dim = attn.head_dim;
        let token_stride = attn.layout.num_kv_heads * 2 * dim;
        for_each_block(
            AttnEntry {
                start,
                pos: end - 1,
                ..entry
            },
            self.block_size,
            |si, off, take, pb| {
                let Some(pb) = pb else {
                    return;
                };
                let base = attn.layout.index(layer, pb, off, kv_h, true);
                for i in 0..take {
                    let value = &kv[base + i * token_stride..][..dim];
                    for g in 0..d {
                        let weight = scores
                            [(g * TILE + row) * attn.score_stride + start - entry.start + si + i];
                        if weight != 0.0 {
                            axpy(&mut values[(g * TILE + row) * dim..][..dim], weight, value);
                        }
                    }
                }
            },
        );
    }
}
