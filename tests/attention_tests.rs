//! Paged attention: the scheduling must not touch the arithmetic.
//!
//! The kernel groups query heads so that a key/value vector is read once and
//! used by every query head that shares it, and resolves the block table once
//! per block instead of once per token. Both are pure scheduling: no operation
//! is added, removed or reordered. So the bar is not "close enough" — it is
//! that the output does not change by one bit, whatever lane width is chosen.
//!
//! Two independent checks enforce that here. First, a deliberately naive
//! reference written in this file — one head at a time, one token at a time,
//! addresses resolved per token — which is the implementation the kernel
//! replaced. Second, a sweep over every legal `heads_per_lane`, which must all
//! agree with each other and with the reference.

use paged_infer::attention::{AttnEntry, PagedAttention};
use paged_infer::memory::allocator::{BlockAllocator, PhysicalBlock};
use paged_infer::memory::block_table::BlockTable;
use paged_infer::memory::layout::KvLayout;

struct Case {
    layout: KvLayout,
    num_heads: usize,
    kv_cache: Vec<f32>,
    tables: Vec<BlockTable>,
    q: Vec<f32>,
}

impl Case {
    /// `blocks_per_seq` blocks handed out in a deliberately scattered order, so
    /// a bug that assumes logically-adjacent blocks are physically adjacent
    /// shows up instead of hiding.
    fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        batch: usize,
        blocks_per_seq: usize,
        mapped_blocks: usize,
    ) -> Self {
        let total = blocks_per_seq * batch + 3;
        let layout = KvLayout::new(1, total, block_size, num_kv_heads, head_dim);
        let mut allocator = BlockAllocator::new(total, block_size);
        let mut handed: Vec<PhysicalBlock> =
            (0..total).map(|_| allocator.allocate().unwrap()).collect();
        handed.reverse();

        let tables = (0..batch)
            .map(|b| {
                let mut t = BlockTable::new();
                // Only `mapped_blocks` of the sequence are backed by physical
                // memory; positions past them must be masked out, not read.
                for i in 0..mapped_blocks {
                    t.append_block(handed[(b * blocks_per_seq + i) % total]);
                }
                t
            })
            .collect();

        let kv_cache = (0..layout.total_floats())
            .map(|i| ((i % 173) as f32) * 0.017 - 1.3)
            .collect();
        let q = (0..batch * num_heads * head_dim)
            .map(|i| ((i % 91) as f32) * 0.031 - 1.1)
            .collect();

        Self {
            layout,
            num_heads,
            kv_cache,
            tables,
            q,
        }
    }

    fn entries(&self, positions: &[usize], window: Option<usize>) -> Vec<AttnEntry<'_>> {
        self.tables
            .iter()
            .zip(positions)
            .map(|(t, &pos)| AttnEntry::new(t, pos, window))
            .collect()
    }

    /// The implementation the kernel replaced: one rayon task per query head,
    /// block table resolved per token, K and V streamed separately by every
    /// head of a group.
    fn reference(&self, q: &[f32], entries: &[AttnEntry<'_>], block_size: usize) -> Vec<f32> {
        let head_dim = self.layout.head_dim;
        let kv_group = self.num_heads / self.layout.num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; entries.len() * self.num_heads * head_dim];

        for (b, entry) in entries.iter().enumerate() {
            for h in 0..self.num_heads {
                let kv_h = h / kv_group;
                let qs = (b * self.num_heads + h) * head_dim;
                let q_head = &q[qs..qs + head_dim];
                let used = entry.window_len();
                let mut scores = vec![0.0f32; used];

                for (si, t) in (entry.start..=entry.pos).enumerate() {
                    scores[si] = match entry.block_table.get_physical_location(t, block_size) {
                        Some((pb, off)) => {
                            let k = self.layout.index(0, pb.index, off, kv_h, false);
                            let dot: f32 = q_head
                                .iter()
                                .zip(&self.kv_cache[k..k + head_dim])
                                .map(|(a, b)| a * b)
                                .sum();
                            dot * scale
                        }
                        None => f32::NEG_INFINITY,
                    };
                }
                paged_infer::math::softmax_in_place(&mut scores);

                let o = (b * self.num_heads + h) * head_dim;
                for (si, t) in (entry.start..=entry.pos).enumerate() {
                    let w = scores[si];
                    if w == 0.0 {
                        continue;
                    }
                    if let Some((pb, off)) = entry.block_table.get_physical_location(t, block_size)
                    {
                        let v = self.layout.index(0, pb.index, off, kv_h, true);
                        for d in 0..head_dim {
                            out[o + d] += w * self.kv_cache[v + d];
                        }
                    }
                }
            }
        }
        out
    }

    /// Run the real kernel at one lane width.
    fn kernel(
        &self,
        q: &[f32],
        entries: &[AttnEntry<'_>],
        block_size: usize,
        heads_per_lane: usize,
        score_stride: usize,
    ) -> Vec<f32> {
        let attn = PagedAttention {
            layout: self.layout,
            block_size,
            num_heads: self.num_heads,
            head_dim: self.layout.head_dim,
            kv_group: self.num_heads / self.layout.num_kv_heads,
            score_stride,
            heads_per_lane,
        };
        let mut out = vec![0.0f32; entries.len() * self.num_heads * self.layout.head_dim];
        let mut scores = vec![0.0f32; entries.len() * self.num_heads * score_stride];
        attn.run(&mut out, &mut scores, q, &self.kv_cache, entries, 0);
        out
    }
}

fn lane_widths(kv_group: usize) -> Vec<usize> {
    (1..=kv_group)
        .filter(|d| kv_group.is_multiple_of(*d))
        .collect()
}

#[test]
fn test_every_lane_width_matches_the_naive_reference_bit_for_bit() {
    // The headline claim. Grouping query heads and walking blocks instead of
    // tokens changes the memory traffic and nothing else, so the result has to
    // be identical — not within a tolerance.
    let block_size = 8;
    for (num_heads, num_kv_heads) in [(8, 1), (8, 2), (8, 4), (8, 8), (6, 3), (12, 4)] {
        let kv_group = num_heads / num_kv_heads;
        for batch in [1usize, 3] {
            let case = Case::new(num_heads, num_kv_heads, 4, block_size, batch, 5, 5);
            // A ragged batch: entries sit at different positions, so the widest
            // window sets the stride and shorter entries use part of a lane.
            let positions: Vec<usize> = (0..batch).map(|b| 39 - b * 9).collect();
            let entries = case.entries(&positions, None);
            let stride = entries.iter().map(|e| e.window_len()).max().unwrap();

            let expected = case.reference(&case.q, &entries, block_size);
            for d in lane_widths(kv_group) {
                let got = case.kernel(&case.q, &entries, block_size, d, stride);
                assert_eq!(
                    got, expected,
                    "heads={num_heads} kv_heads={num_kv_heads} batch={batch} \
                     heads_per_lane={d} diverged from the reference"
                );
            }
        }
    }
}

#[test]
fn test_lane_widths_agree_across_block_boundaries_and_window_sizes() {
    // A window can start mid-block, end mid-block, or span exactly one block.
    // Each is a chance for the block walk to drop or double-count a position.
    let block_size = 8;
    let case = Case::new(8, 2, 4, block_size, 1, 6, 6);
    for pos in 0..48 {
        for window in [
            None,
            Some(1),
            Some(2),
            Some(7),
            Some(8),
            Some(9),
            Some(16),
            Some(33),
        ] {
            let entries = case.entries(&[pos], window);
            let stride = entries[0].window_len().max(1);
            let expected = case.reference(&case.q, &entries, block_size);
            for d in lane_widths(4) {
                assert_eq!(
                    case.kernel(&case.q, &entries, block_size, d, stride),
                    expected,
                    "pos {pos} window {window:?} heads_per_lane {d}"
                );
            }
        }
    }
}

#[test]
fn test_unmapped_positions_are_masked_not_read() {
    // Only 2 of 6 logical blocks are mapped. Everything past them must be
    // -inf-masked; a zero score would survive the softmax and pull garbage
    // from whatever physical block the arithmetic happened to land on.
    let block_size = 8;
    let case = Case::new(8, 2, 4, block_size, 1, 6, 2);
    let entries = case.entries(&[40], None);
    let stride = entries[0].window_len();

    let expected = case.reference(&case.q, &entries, block_size);
    for d in lane_widths(4) {
        assert_eq!(
            case.kernel(&case.q, &entries, block_size, d, stride),
            expected
        );
    }

    // And the answer must equal attending to just the mapped span, because the
    // rest contributes exactly nothing.
    let truncated = case.entries(&[15], None);
    assert_eq!(
        case.kernel(&case.q, &entries, block_size, 4, stride),
        case.kernel(&case.q, &truncated, block_size, 4, 16),
        "masked-out positions changed the result"
    );
}

#[test]
fn test_a_sliding_window_narrower_than_one_block() {
    // The whole window lives inside a single block and starts at an arbitrary
    // offset, so the block walk's first-and-last-partial handling is the only
    // thing being exercised.
    let block_size = 16;
    let case = Case::new(8, 4, 4, block_size, 1, 4, 4);
    for pos in [0usize, 1, 15, 16, 17, 31, 47] {
        for window in [Some(1), Some(2), Some(3)] {
            let entries = case.entries(&[pos], window);
            let stride = entries[0].window_len();
            let expected = case.reference(&case.q, &entries, block_size);
            for d in lane_widths(2) {
                assert_eq!(
                    case.kernel(&case.q, &entries, block_size, d, stride),
                    expected,
                    "pos {pos} window {window:?} d {d}"
                );
            }
        }
    }
}

#[test]
fn test_entries_sharing_one_block_table_stay_independent() {
    // Batched prefill and speculative decoding both put several positions of
    // ONE sequence into one batch, so the same &BlockTable appears more than
    // once. Each entry must still attend only up to its own position.
    let block_size = 8;
    // Three batch entries' worth of queries, but every entry pointed at the
    // same sequence's block table.
    let case = Case::new(8, 2, 4, block_size, 3, 5, 5);
    let positions = [7usize, 19, 33];
    let shared: Vec<AttnEntry<'_>> = positions
        .iter()
        .map(|&pos| AttnEntry::new(&case.tables[0], pos, None))
        .collect();
    let stride = shared.iter().map(|e| e.window_len()).max().unwrap();

    let expected = case.reference(&case.q, &shared, block_size);
    for d in lane_widths(4) {
        assert_eq!(
            case.kernel(&case.q, &shared, block_size, d, stride),
            expected
        );
    }

    // Each entry must match what it would have produced on its own — proof
    // that batching one sequence's positions together did not leak state
    // between them.
    let head_dim = case.layout.head_dim;
    let per_entry = case.num_heads * head_dim;
    for (i, &pos) in positions.iter().enumerate() {
        let alone = case.entries(&[pos], None);
        let solo = case.kernel(
            &case.q[i * per_entry..(i + 1) * per_entry],
            &alone,
            block_size,
            2,
            alone[0].window_len(),
        );
        assert_eq!(
            &expected[i * per_entry..(i + 1) * per_entry],
            &solo[..],
            "entry at position {pos} changed when batched with its siblings"
        );
    }
}

#[test]
fn test_a_zero_width_window_attends_to_nothing_rather_than_panicking() {
    // `Some(0)` is a meaningless window, but it is reachable from a config file
    // and it used to reach rayon as `par_chunks_mut(0)` — a panic inside a
    // worker thread whose message says nothing about the configuration that
    // caused it. The stride is floored at one slot per lane instead, and the
    // right answer for attending to nothing is all zeros.
    let block_size = 8;
    let case = Case::new(8, 2, 4, block_size, 1, 4, 4);
    let entry = AttnEntry::new(&case.tables[0], 20, Some(0));
    assert_eq!(entry.window_len(), 0);

    let out = case.kernel(&case.q, &[entry], block_size, 2, 1);
    assert!(
        out.iter().all(|v| *v == 0.0),
        "a zero-width window should contribute nothing"
    );
}

#[test]
fn test_a_score_stride_wider_than_the_window_is_harmless() {
    // The batch sizes the score buffer to the widest window, so a short entry
    // gets a lane with unused tail slots holding whatever was there before.
    // Those must never be read.
    let block_size = 8;
    let case = Case::new(8, 2, 4, block_size, 2, 5, 5);
    let entries = case.entries(&[9, 34], None);
    let tight = entries.iter().map(|e| e.window_len()).max().unwrap();
    let expected = case.reference(&case.q, &entries, block_size);

    for stride in [tight, tight + 1, tight + 17, tight * 2] {
        for d in lane_widths(4) {
            assert_eq!(
                case.kernel(&case.q, &entries, block_size, d, stride),
                expected,
                "stride {stride} d {d} read past the window"
            );
        }
    }
}
