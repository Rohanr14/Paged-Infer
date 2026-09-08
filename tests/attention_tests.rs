//! Paged attention: the scheduling must not touch the arithmetic.
//!
//! The kernel groups query heads so that a key/value vector is read once and
//! used by every query head that shares it, and resolves the block table once
//! per block instead of once per token. Both are pure scheduling: no operation
//! is added, removed or reordered. So the bar for *that* claim is not "close
//! enough" — it is that the output does not change by one bit, whatever lane
//! width is chosen.
//!
//! Two different references enforce two different properties here:
//!
//! * **Same arithmetic, naive schedule** — one head at a time, one token at a
//!   time, addresses resolved per token, but built from the production `dot`,
//!   `softmax_in_place` and `axpy` primitives. This is exactly the
//!   implementation the kernel replaced, and every lane width must match it
//!   bit for bit. It is what proves the kernel is a rescheduling and nothing
//!   more.
//!
//! * **Independent oracle** — the attention equation written from scratch in
//!   f64 with plain scalar sums. It shares no code with the kernel, so it
//!   catches an arithmetic bug the first reference would faithfully reproduce
//!   (a wrong scale, a dropped softmax subtraction, a fused multiply-add that
//!   silently changed semantics). The price of independence is that it cannot
//!   be compared exactly: the kernel accumulates in f32 across vector lanes
//!   and, on NEON and AVX2+FMA, with fused multiply-adds, and each of those
//!   rounds differently from a sequential f64 sum. Comparing them bit for bit
//!   asserts something the oracle cannot provide, and fails on any machine
//!   whose kernel is not the scalar tail loop — which is why the comparison is
//!   a tolerance, with a finiteness check so NaN can never pass.

use paged_infer::attention::{AttnEntry, PagedAttention};
use paged_infer::math::{axpy, dot, softmax_in_place};
use paged_infer::memory::allocator::{BlockAllocator, PhysicalBlock};
use paged_infer::memory::block_table::BlockTable;
use paged_infer::memory::layout::KvLayout;

/// How far the f32 kernel may sit from the f64 oracle.
///
/// Every f32 operation rounds by at most half an ulp, `2^-24 ≈ 6e-8` relative.
/// The largest case below is a 128-wide dot product over operands of magnitude
/// below 2, so the unscaled score carries at most ~128 roundings on partial
/// sums below ~300 — under `2e-3` worst case, `~1e-4` typical, and it is then
/// scaled by `1/sqrt(128)`. The softmax turns a score error `δ` into a relative
/// weight error of about `δ`, and the value sum over at most 64 tokens of
/// magnitude below 2 adds at most 64 more roundings. That bounds the kernel's
/// deviation from exact arithmetic well under `1e-4` absolute on outputs of
/// magnitude below 2 — while a scheduling bug (a key, value, head, block or
/// position read from the wrong place) moves the output by `O(0.01)` or more.
/// The measured deviation on x86 is printed by the tests and sits two orders
/// of magnitude under the bound.
const ORACLE_ABS_TOL: f32 = 1e-4;
const ORACLE_REL_TOL: f32 = 1e-4;

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

    fn kv_group(&self) -> usize {
        self.num_heads / self.layout.num_kv_heads
    }

    /// The implementation the kernel replaced: one task per query head, block
    /// table resolved per token, K and V streamed separately by every head of
    /// a group — but the same `dot`, `softmax_in_place` and `axpy` the kernel
    /// uses, applied in the same order. Bit-identical to the kernel by
    /// construction.
    fn same_arithmetic(&self, q: &[f32], entries: &[AttnEntry<'_>], block_size: usize) -> Vec<f32> {
        let head_dim = self.layout.head_dim;
        let kv_group = self.kv_group();
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
                            dot(q_head, &self.kv_cache[k..k + head_dim]) * scale
                        }
                        None => f32::NEG_INFINITY,
                    };
                }
                softmax_in_place(&mut scores);

                let o = (b * self.num_heads + h) * head_dim;
                for (si, t) in (entry.start..=entry.pos).enumerate() {
                    let w = scores[si];
                    if w == 0.0 {
                        continue;
                    }
                    if let Some((pb, off)) = entry.block_table.get_physical_location(t, block_size)
                    {
                        let v = self.layout.index(0, pb.index, off, kv_h, true);
                        axpy(
                            &mut out[o..o + head_dim],
                            w,
                            &self.kv_cache[v..v + head_dim],
                        );
                    }
                }
            }
        }
        out
    }

    /// The attention equation from scratch, in f64, sharing nothing with the
    /// kernel. Unmapped positions simply do not take part, which is what the
    /// kernel's `-inf` mask means.
    fn oracle(&self, q: &[f32], entries: &[AttnEntry<'_>], block_size: usize) -> Vec<f32> {
        let head_dim = self.layout.head_dim;
        let kv_group = self.kv_group();
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut out = vec![0.0f32; entries.len() * self.num_heads * head_dim];

        for (b, entry) in entries.iter().enumerate() {
            for h in 0..self.num_heads {
                let kv_h = h / kv_group;
                let qs = (b * self.num_heads + h) * head_dim;
                let q_head = &q[qs..qs + head_dim];

                let mut mapped = Vec::new();
                let mut scores = Vec::new();
                for t in entry.start..=entry.pos {
                    if let Some((pb, off)) = entry.block_table.get_physical_location(t, block_size)
                    {
                        let k = self.layout.index(0, pb.index, off, kv_h, false);
                        let s: f64 = q_head
                            .iter()
                            .zip(&self.kv_cache[k..k + head_dim])
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum();
                        mapped.push((pb.index, off));
                        scores.push(s * scale);
                    }
                }
                if scores.is_empty() {
                    continue;
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let denom: f64 = weights.iter().sum();

                let mut acc = vec![0.0f64; head_dim];
                for (w, &(pb, off)) in weights.iter().zip(&mapped) {
                    let v = self.layout.index(0, pb, off, kv_h, true);
                    for (a, x) in acc.iter_mut().zip(&self.kv_cache[v..v + head_dim]) {
                        *a += (w / denom) * *x as f64;
                    }
                }
                let o = (b * self.num_heads + h) * head_dim;
                for (slot, a) in out[o..o + head_dim].iter_mut().zip(&acc) {
                    *slot = *a as f32;
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
            kv_group: self.kv_group(),
            score_stride,
            heads_per_lane,
        };
        let mut out = vec![0.0f32; entries.len() * self.num_heads * self.layout.head_dim];
        let mut scores = vec![0.0f32; entries.len() * self.num_heads * score_stride];
        attn.run(&mut out, &mut scores, q, &self.kv_cache, entries, 0);
        out
    }

    /// Both properties, for every legal lane width, in one place: exact
    /// agreement with the same-arithmetic schedule (and therefore with every
    /// other lane width), and bounded deviation from the independent oracle.
    /// Returns the worst deviation from the oracle seen.
    fn check_all_lanes(
        &self,
        entries: &[AttnEntry<'_>],
        block_size: usize,
        score_stride: usize,
        ctx: &str,
    ) -> f32 {
        let exact = self.same_arithmetic(&self.q, entries, block_size);
        let oracle = self.oracle(&self.q, entries, block_size);
        let mut worst = 0.0f32;
        for d in lane_widths(self.kv_group()) {
            let got = self.kernel(&self.q, entries, block_size, d, score_stride);
            assert_eq!(
                got, exact,
                "{ctx} heads_per_lane={d}: the kernel is not a pure rescheduling \
                 of the per-token loop"
            );
            worst = worst.max(assert_close(
                &got,
                &oracle,
                &format!("{ctx} heads_per_lane={d}"),
            ));
        }
        worst
    }
}

fn lane_widths(kv_group: usize) -> Vec<usize> {
    (1..=kv_group)
        .filter(|d| kv_group.is_multiple_of(*d))
        .collect()
}

/// Every element finite and within the documented tolerance of the oracle.
/// Returns the largest absolute deviation.
fn assert_close(got: &[f32], oracle: &[f32], ctx: &str) -> f32 {
    assert_eq!(got.len(), oracle.len(), "{ctx}: output length");
    let mut worst = 0.0f32;
    for (i, (g, e)) in got.iter().zip(oracle).enumerate() {
        assert!(g.is_finite(), "{ctx}: element {i} is {g}");
        let allowed = ORACLE_ABS_TOL + ORACLE_REL_TOL * e.abs();
        let dev = (g - e).abs();
        assert!(
            dev <= allowed,
            "{ctx}: element {i}: kernel {g} vs oracle {e} (|delta|={dev:e}, allowed {allowed:e})"
        );
        worst = worst.max(dev);
    }
    worst
}

#[test]
fn test_every_lane_width_matches_the_per_token_schedule_bit_for_bit() {
    // The headline claim. Grouping query heads and walking blocks instead of
    // tokens changes the memory traffic and nothing else, so the result has to
    // be identical to the per-token loop built from the same primitives — not
    // within a tolerance. The f64 oracle rides along at its own tolerance.
    let block_size = 8;
    for head_dim in [4usize, 64] {
        for (num_heads, num_kv_heads) in [(8, 1), (8, 2), (8, 4), (8, 8), (6, 3), (12, 4)] {
            for batch in [1usize, 3] {
                let case = Case::new(num_heads, num_kv_heads, head_dim, block_size, batch, 5, 5);
                // A ragged batch: entries sit at different positions, so the
                // widest window sets the stride and shorter entries use part
                // of a lane.
                let positions: Vec<usize> = (0..batch).map(|b| 39 - b * 9).collect();
                let entries = case.entries(&positions, None);
                let stride = entries.iter().map(|e| e.window_len()).max().unwrap();
                case.check_all_lanes(
                    &entries,
                    block_size,
                    stride,
                    &format!(
                        "head_dim={head_dim} heads={num_heads} kv_heads={num_kv_heads} batch={batch}"
                    ),
                );
            }
        }
    }
}

#[test]
fn test_kernel_tracks_the_independent_oracle_at_realistic_head_dims() {
    // 64 and 128 are what real checkpoints use; the others are chosen to leave
    // every kind of tail the vector kernels have — below one vector, a partial
    // unrolled chunk, a whole chunk plus a few scalars — so a kernel that drops
    // or double-counts its remainder cannot hide behind a convenient width.
    let block_size = 16;
    let mut worst = 0.0f32;
    for head_dim in [4usize, 20, 36, 64, 72, 100, 128] {
        for (num_heads, num_kv_heads) in [(8, 2), (4, 4), (8, 1)] {
            let case = Case::new(num_heads, num_kv_heads, head_dim, block_size, 2, 4, 4);
            let entries = case.entries(&[63, 40], None);
            let stride = entries.iter().map(|e| e.window_len()).max().unwrap();
            let dev = case.check_all_lanes(
                &entries,
                block_size,
                stride,
                &format!("head_dim={head_dim} heads={num_heads} kv_heads={num_kv_heads}"),
            );
            worst = worst.max(dev);
        }
    }
    println!(
        "worst |kernel - f64 oracle| = {worst:e} (allowed {ORACLE_ABS_TOL:e} + {ORACLE_REL_TOL:e}*|x|)"
    );
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
            case.check_all_lanes(
                &entries,
                block_size,
                stride,
                &format!("pos {pos} window {window:?}"),
            );
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
    case.check_all_lanes(&entries, block_size, stride, "partially mapped");

    // And the answer must equal attending to just the mapped span, because the
    // rest contributes exactly nothing: a masked score becomes an exact 0.0
    // weight, which adds nothing to the softmax denominator and is skipped in
    // the value accumulation, so this comparison is exact.
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
            case.check_all_lanes(
                &entries,
                block_size,
                stride,
                &format!("pos {pos} window {window:?}"),
            );
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
    case.check_all_lanes(&shared, block_size, stride, "shared table");

    // Each entry must match what it would have produced on its own — proof
    // that batching one sequence's positions together did not leak state
    // between them. Same arithmetic either way, so exact.
    let batched = case.kernel(&case.q, &shared, block_size, 2, stride);
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
            &batched[i * per_entry..(i + 1) * per_entry],
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

    for stride in [tight, tight + 1, tight + 17, tight * 2] {
        case.check_all_lanes(&entries, block_size, stride, &format!("stride {stride}"));
    }
}
