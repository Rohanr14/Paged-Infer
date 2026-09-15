//! Cross-request prefix reuse must preserve the full-window attention equation.
//!
//! Compare bitwise with the existing schedule and the same arithmetic performed
//! per token, then check against an independent scalar f64 oracle. The oracle
//! tolerance covers f32 SIMD reduction/FMA rounding and checks the attention
//! equation; the separate bitwise comparison checks arithmetic preservation.

use paged_infer::attention::{AttnEntry, PagedAttention, SharedPrefixPlan, SharedPrefixScratch};
use paged_infer::math::{axpy, dot, softmax_in_place};
use paged_infer::memory::allocator::{BlockAllocator, PhysicalBlock};
use paged_infer::memory::block_table::BlockTable;
use paged_infer::memory::layout::KvLayout;

struct Case {
    layout: KvLayout,
    num_heads: usize,
    block_size: usize,
    allocator: BlockAllocator,
    blocks: Vec<PhysicalBlock>,
    tables: Vec<BlockTable>,
    kv: Vec<f32>,
    q: Vec<f32>,
}

impl Case {
    /// Equal labels denote the same live physical block. Reverse the allocated
    /// blocks so that logical neighbors never imply adjacent physical addresses.
    fn new(
        kv_heads: usize,
        group: usize,
        head_dim: usize,
        block_size: usize,
        mappings: &[&[usize]],
    ) -> Self {
        let total = mappings
            .iter()
            .flat_map(|m| m.iter())
            .max()
            .copied()
            .unwrap_or(0)
            + 4;
        let layout = KvLayout::new(2, total, block_size, kv_heads, head_dim);
        let mut allocator = BlockAllocator::new(total, block_size);
        let blocks: Vec<_> = (0..total)
            .map(|_| allocator.allocate().unwrap())
            .rev()
            .collect();
        let mut refs = vec![0; total];
        let tables = mappings
            .iter()
            .map(|mapping| {
                let mut table = BlockTable::new();
                for &label in *mapping {
                    if refs[label] != 0 {
                        allocator.incref(blocks[label]);
                    }
                    refs[label] += 1;
                    table.append_block(blocks[label]);
                }
                table
            })
            .collect();
        for (label, &references) in refs.iter().enumerate() {
            if references == 0 {
                allocator.free(blocks[label]);
            }
        }
        let kv = (0..layout.total_floats())
            .map(|i| ((i * 37 + i / 17) % 257) as f32 * 0.009 - 1.15)
            .collect();
        let num_heads = kv_heads * group;
        let q = (0..mappings.len() * num_heads * head_dim)
            .map(|i| ((i * 19 + i / 11) % 127) as f32 * 0.017 - 0.95)
            .collect();
        Self {
            layout,
            num_heads,
            block_size,
            allocator,
            blocks,
            tables,
            kv,
            q,
        }
    }

    fn entries(&self, positions: &[usize], starts: &[usize]) -> Vec<AttnEntry<'_>> {
        assert_eq!(positions.len(), self.tables.len());
        assert_eq!(starts.len(), self.tables.len());
        self.tables
            .iter()
            .zip(positions)
            .zip(starts)
            .map(|((table, &pos), &start)| AttnEntry {
                block_table: table,
                pos,
                start,
            })
            .collect()
    }

    fn same_arithmetic(&self, entries: &[AttnEntry<'_>], layer: usize) -> Vec<f32> {
        let dim = self.layout.head_dim;
        let scale = 1.0 / (dim as f32).sqrt();
        let mut result = vec![0.0; self.q.len()];
        for (row, entry) in entries.iter().enumerate() {
            for head in 0..self.num_heads {
                let kv_head = head / (self.num_heads / self.layout.num_kv_heads);
                let offset = (row * self.num_heads + head) * dim;
                let q = &self.q[offset..offset + dim];
                let mut scores = vec![f32::NEG_INFINITY; entry.window_len()];
                for (i, token) in (entry.start..=entry.pos).enumerate() {
                    if let Some((block, slot)) = entry
                        .block_table
                        .get_physical_location(token, self.block_size)
                    {
                        let key = self.layout.index(layer, block.index, slot, kv_head, false);
                        scores[i] = dot(q, &self.kv[key..key + dim]) * scale;
                    }
                }
                softmax_in_place(&mut scores);
                for (i, token) in (entry.start..=entry.pos).enumerate() {
                    if scores[i] == 0.0 {
                        continue;
                    }
                    if let Some((block, slot)) = entry
                        .block_table
                        .get_physical_location(token, self.block_size)
                    {
                        let value = self.layout.index(layer, block.index, slot, kv_head, true);
                        axpy(
                            &mut result[offset..offset + dim],
                            scores[i],
                            &self.kv[value..value + dim],
                        );
                    }
                }
            }
        }
        result
    }

    fn oracle(&self, entries: &[AttnEntry<'_>], layer: usize) -> Vec<f32> {
        let dim = self.layout.head_dim;
        let mut result = vec![0.0; self.q.len()];
        for (row, entry) in entries.iter().enumerate() {
            for head in 0..self.num_heads {
                let kv_head = head / (self.num_heads / self.layout.num_kv_heads);
                let offset = (row * self.num_heads + head) * dim;
                let q = &self.q[offset..offset + dim];
                let mut positions = Vec::new();
                let mut scores = Vec::new();
                for token in entry.start..=entry.pos {
                    if let Some((block, slot)) = entry
                        .block_table
                        .get_physical_location(token, self.block_size)
                    {
                        let key = self.layout.index(layer, block.index, slot, kv_head, false);
                        let score: f64 = q
                            .iter()
                            .zip(&self.kv[key..key + dim])
                            .map(|(&a, &b)| a as f64 * b as f64)
                            .sum();
                        positions.push((block.index, slot));
                        scores.push(score / (dim as f64).sqrt());
                    }
                }
                if scores.is_empty() {
                    continue;
                }
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
                let denominator: f64 = weights.iter().sum();
                let mut sum = vec![0.0f64; dim];
                for (&weight, &(block, slot)) in weights.iter().zip(&positions) {
                    let value = self.layout.index(layer, block, slot, kv_head, true);
                    for (acc, &v) in sum.iter_mut().zip(&self.kv[value..value + dim]) {
                        *acc += weight / denominator * v as f64;
                    }
                }
                for (out, acc) in result[offset..offset + dim].iter_mut().zip(sum) {
                    *out = acc as f32;
                }
            }
        }
        result
    }

    /// Exercise all legal existing head-lane widths and both KV layers. Reuse
    /// dirty scratch/output buffers so missed initialization cannot hide in zeros.
    fn check(
        &self,
        positions: &[usize],
        starts: &[usize],
        shared_tokens: Option<usize>,
        scratch: &mut SharedPrefixScratch,
    ) {
        let entries = self.entries(positions, starts);
        let plan = SharedPrefixPlan::new(&entries, self.block_size);
        assert_eq!(plan.as_ref().map(|p| p.prefix_tokens()), shared_tokens);
        let group = self.num_heads / self.layout.num_kv_heads;
        let stride = entries
            .iter()
            .map(|entry| entry.window_len())
            .max()
            .unwrap_or(0)
            + 5;
        for layer in 0..self.layout.num_layers {
            let reference = self.same_arithmetic(&entries, layer);
            let oracle = self.oracle(&entries, layer);
            for lanes in (1..=group).filter(|lanes| group.is_multiple_of(*lanes)) {
                let attn = PagedAttention {
                    layout: self.layout,
                    block_size: self.block_size,
                    num_heads: self.num_heads,
                    head_dim: self.layout.head_dim,
                    kv_group: group,
                    score_stride: stride,
                    heads_per_lane: lanes,
                };
                let mut baseline = vec![f32::NAN; self.q.len()];
                let mut scores = vec![f32::NAN; entries.len() * self.num_heads * stride];
                attn.run(
                    &mut baseline,
                    &mut scores,
                    &self.q,
                    &self.kv,
                    &entries,
                    layer,
                );
                let mut output = vec![f32::NAN; self.q.len()];
                if let Some(plan) = &plan {
                    plan.run(&attn, &mut output, &self.q, &self.kv, layer, scratch);
                } else {
                    attn.run(&mut output, &mut scores, &self.q, &self.kv, &entries, layer);
                }
                for (i, ((&actual, &expected), (&old, &independent))) in output
                    .iter()
                    .zip(&reference)
                    .zip(baseline.iter().zip(&oracle))
                    .enumerate()
                {
                    assert!(
                        actual.is_finite(),
                        "nonfinite output {i}, layer {layer}, lane {lanes}"
                    );
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "arithmetic changed at {i}, layer {layer}, lane {lanes}"
                    );
                    assert_eq!(
                        actual.to_bits(),
                        old.to_bits(),
                        "baseline differs at {i}, layer {layer}, lane {lanes}"
                    );
                    let tolerance = 1e-4 + 1e-4 * independent.abs();
                    assert!((actual - independent).abs() <= tolerance,
                        "oracle differs at {i}, layer {layer}, lane {lanes}: {actual} vs {independent}");
                }
            }
        }
    }
}

#[test]
fn shared_prefix_and_private_suffix_match_both_references() {
    let mut scratch = SharedPrefixScratch::default();
    // Batch widths cover the four-row path, two-row path, and single-row tail;
    // GQA and dimensions cover head grouping, SIMD widths, and scalar tails.
    for batch in [2, 3, 4, 5, 7] {
        let mappings: Vec<Vec<_>> = (0..batch)
            .map(|row| vec![0, 1, 2 + row * 2, 3 + row * 2])
            .collect();
        let views: Vec<_> = mappings.iter().map(Vec::as_slice).collect();
        for (kv_heads, group, dim) in [(1, 1, 1), (2, 2, 7), (2, 3, 17), (1, 8, 33), (2, 4, 128)] {
            let case = Case::new(kv_heads, group, dim, 4, &views);
            case.check(&vec![13; batch], &vec![0; batch], Some(8), &mut scratch);
        }
    }
}

#[test]
fn ragged_windows_keep_private_leading_and_trailing_tokens_in_one_softmax() {
    let case = Case::new(
        2,
        4,
        19,
        4,
        &[&[0, 1, 2, 3, 4], &[0, 1, 2, 5], &[0, 1, 2, 6, 7]],
    );
    // Shared work starts halfway through block 1. Earlier attended tokens are
    // still private work for rows 0/1, and suffix lengths are different.
    case.check(
        &[18, 14, 16],
        &[0, 3, 6],
        Some(6),
        &mut SharedPrefixScratch::default(),
    );
}

#[test]
fn unmapped_private_suffix_is_masked_after_the_common_prefix() {
    let case = Case::new(1, 4, 31, 3, &[&[0, 1], &[0, 1, 2], &[0, 1, 3, 4]]);
    case.check(
        &[11, 9, 13],
        &[0, 1, 2],
        Some(4),
        &mut SharedPrefixScratch::default(),
    );
}

#[test]
fn current_write_block_is_excluded_even_when_its_mapping_is_shared() {
    let case = Case::new(2, 2, 16, 4, &[&[0, 1, 2], &[0, 1, 2]]);
    let mut scratch = SharedPrefixScratch::default();
    case.check(&[7, 10], &[0, 0], Some(4), &mut scratch);
    case.check(&[8, 10], &[0, 0], Some(8), &mut scratch);
    case.check(&[3, 9], &[0, 0], None, &mut scratch);
}

#[test]
fn no_sharing_and_single_request_use_the_existing_path() {
    let mut scratch = SharedPrefixScratch::default();
    let single = Case::new(1, 4, 9, 4, &[&[0, 1, 2]]);
    single.check(&[10], &[0], None, &mut scratch);
    let distinct = Case::new(1, 4, 9, 4, &[&[0, 1, 2], &[3, 4, 5]]);
    distinct.check(&[10, 10], &[0, 0], None, &mut scratch);
    let empty: Vec<AttnEntry<'_>> = Vec::new();
    assert!(SharedPrefixPlan::new(&empty, 4).is_none());
}

#[test]
fn a_physical_block_at_a_different_logical_position_is_not_a_prefix() {
    let case = Case::new(1, 2, 15, 4, &[&[0, 1, 2, 3], &[1, 0, 4, 5]]);
    case.check(
        &[14, 14],
        &[0, 0],
        None,
        &mut SharedPrefixScratch::default(),
    );
    // A later matching block does not restart prefix sharing after divergence.
    let diverged = Case::new(1, 2, 15, 4, &[&[0, 1, 2, 3], &[0, 4, 2, 5]]);
    diverged.check(
        &[14, 14],
        &[0, 0],
        Some(4),
        &mut SharedPrefixScratch::default(),
    );
}

#[test]
fn disjoint_windows_and_entirely_masked_rows_decline_prefix_reuse() {
    let case = Case::new(1, 2, 11, 4, &[&[0, 1, 2], &[0, 1, 3]]);
    let mut scratch = SharedPrefixScratch::default();
    case.check(&[10, 10], &[0, 8], None, &mut scratch);
    case.check(&[10, 10], &[0, 11], None, &mut scratch);
    // The second row's entire nonempty window lies beyond its mapped table.
    let unmapped = Case::new(1, 2, 11, 4, &[&[0, 1, 2], &[0, 1]]);
    unmapped.check(&[10, 14], &[0, 12], None, &mut scratch);
}

#[test]
fn fresh_plan_observes_cow_remapping_and_recycled_physical_blocks() {
    let mut case = Case::new(1, 4, 17, 4, &[&[0, 1, 2], &[0, 1, 3]]);
    let mut scratch = SharedPrefixScratch::default();
    case.check(&[10, 10], &[0, 0], Some(8), &mut scratch);
    let copied = case.allocator.allocate().unwrap();
    let shared = case.blocks[1];
    assert!(case.allocator.is_shared(shared));
    for layer in 0..case.layout.num_layers {
        for slot in 0..case.block_size {
            for is_value in [false, true] {
                let from = case.layout.index(layer, shared.index, slot, 0, is_value);
                let to = case.layout.index(layer, copied.index, slot, 0, is_value);
                case.kv.copy_within(from..from + case.layout.head_dim, to);
            }
        }
    }
    case.tables[1].remap(1, copied);
    assert!(!case.allocator.free(shared));
    case.check(&[10, 10], &[0, 0], Some(4), &mut scratch);

    // The old physical prefix becomes free, then gets reused with different KV
    // data. No cached plan/scratch may retain the former block's identity/data.
    let original = case.blocks[0];
    let replacement = case.allocator.allocate().unwrap();
    case.tables[0].remap(0, replacement);
    assert!(!case.allocator.free(original));
    case.tables[1].remap(0, replacement);
    case.allocator.incref(replacement);
    assert!(case.allocator.free(original));
    let recycled = loop {
        let block = case.allocator.allocate().unwrap();
        if block == original {
            break block;
        }
    };
    case.tables[0].remap(0, recycled);
    assert!(!case.allocator.free(replacement));
    case.tables[1].remap(0, recycled);
    case.allocator.incref(recycled);
    assert!(case.allocator.free(replacement));
    for layer in 0..case.layout.num_layers {
        for slot in 0..case.block_size {
            for is_value in [false, true] {
                let offset = case.layout.index(layer, recycled.index, slot, 0, is_value);
                case.kv[offset..offset + case.layout.head_dim].fill(if is_value {
                    0.75
                } else {
                    -0.5
                });
            }
        }
    }
    case.check(&[10, 10], &[0, 0], Some(4), &mut scratch);
}

#[test]
fn large_finite_scores_keep_softmax_stable_across_shared_and_private_ranges() {
    let mut case = Case::new(
        1,
        2,
        33,
        4,
        &[&[0, 1, 2], &[0, 1, 3], &[0, 1, 4], &[0, 1, 5], &[0, 1, 6]],
    );
    for (i, q) in case.q.iter_mut().enumerate() {
        *q = if (i / case.layout.head_dim).is_multiple_of(2) {
            128.0
        } else {
            -128.0
        };
    }
    for layer in 0..case.layout.num_layers {
        for block in 0..case.layout.num_blocks {
            for slot in 0..case.block_size {
                let key = case.layout.index(layer, block, slot, 0, false);
                let sign = if (block + slot + layer).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                case.kv[key..key + case.layout.head_dim].fill(sign * 128.0);
            }
        }
    }
    case.check(
        &[10; 5],
        &[0, 1, 2, 3, 4],
        Some(4),
        &mut SharedPrefixScratch::default(),
    );
}

fn zero_probability_case(masked_value: f32, underflow: bool) {
    let mut case = Case::new(1, 2, 8, 4, &[&[0, 1], &[0, 2]]);
    case.kv.fill(0.0);
    case.q.fill(0.0);
    for query in case.q.chunks_exact_mut(8) {
        query[0] = 1.0;
    }
    for layer in 0..case.layout.num_layers {
        for token in 0..4 {
            let k = case
                .layout
                .index(layer, case.blocks[0].index, token, 0, false);
            let v = case
                .layout
                .index(layer, case.blocks[0].index, token, 0, true);
            let score = if token == 0 {
                if underflow {
                    -100.0
                } else {
                    0.0
                }
            } else {
                -200.0
            };
            case.kv[k] = score * 8.0f32.sqrt();
            case.kv[v..v + 8].fill(if token == 0 {
                if underflow {
                    -1e-4
                } else {
                    0.5
                }
            } else {
                masked_value
            });
        }
        for block in [case.blocks[1], case.blocks[2]] {
            let v = case.layout.index(layer, block.index, 0, 0, true);
            case.kv[v..v + 8].fill(if underflow { -0.0 } else { 1.0 });
        }
    }
    let entries = case.entries(&[4, 4], &[0, 0]);
    let plan = SharedPrefixPlan::new(&entries, 4).unwrap();
    let attn = PagedAttention {
        layout: case.layout,
        block_size: 4,
        num_heads: 2,
        head_dim: 8,
        kv_group: 2,
        score_stride: 5,
        heads_per_lane: 2,
    };
    let mut scratch = SharedPrefixScratch::default();
    for layer in 0..case.layout.num_layers {
        // f64 would not underflow at these scores: this specifically tests the
        // production f32 probability mask, including signed zeros and NaN V.
        let expected = case.same_arithmetic(&entries, layer);
        let mut actual = vec![f32::NAN; case.q.len()];
        plan.run(&attn, &mut actual, &case.q, &case.kv, layer, &mut scratch);
        assert!(expected.iter().all(|v| v.is_finite()));
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "zero-probability updates must be skipped"
            );
        }
    }
}

#[test]
fn zero_probabilities_preserve_signed_zero_after_finite_underflow() {
    zero_probability_case(1.0, true);
}

#[test]
fn zero_probabilities_do_not_read_masked_nonfinite_values() {
    for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        zero_probability_case(value, false);
    }
}
