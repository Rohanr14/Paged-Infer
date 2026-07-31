//! Physical addressing for the paged KV cache.
//!
//! The cache is one flat `Vec<f32>` indexed as
//!
//! ```text
//! [layer][physical_block][token_in_block][kv_head][k|v][head_dim]
//! ```
//!
//! Putting `layer` outermost means all of one physical block's state for a
//! single layer is contiguous, which is what makes a copy-on-write block clone
//! `num_layers` memcpys instead of one per token.

use std::ops::Range;

use super::allocator::PhysicalBlock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    pub num_layers: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KvLayout {
    pub fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        }
    }

    /// Total `f32` slots a cache with this layout occupies.
    pub fn total_floats(&self) -> usize {
        self.num_layers * self.num_blocks * self.block_size * self.num_kv_heads * 2 * self.head_dim
    }

    /// Slots occupied by one physical block within one layer.
    #[inline]
    pub fn block_stride(&self) -> usize {
        self.block_size * self.num_kv_heads * 2 * self.head_dim
    }

    /// Offset of a single key or value vector.
    #[inline]
    pub fn index(
        &self,
        layer: usize,
        phys_block: usize,
        token_offset: usize,
        kv_head: usize,
        is_value: bool,
    ) -> usize {
        let kv = usize::from(is_value);
        ((((layer * self.num_blocks + phys_block) * self.block_size + token_offset)
            * self.num_kv_heads
            + kv_head)
            * 2
            + kv)
            * self.head_dim
    }

    /// Contiguous span holding one physical block's state for one layer.
    #[inline]
    pub fn block_span(&self, layer: usize, phys_block: usize) -> Range<usize> {
        let start = (layer * self.num_blocks + phys_block) * self.block_stride();
        start..start + self.block_stride()
    }

    /// Duplicate every layer's state for one physical block. This is the data
    /// movement behind a copy-on-write split of a shared block.
    pub fn copy_block(&self, cache: &mut [f32], src: PhysicalBlock, dst: PhysicalBlock) {
        assert_ne!(src.index, dst.index, "copy_block called with src == dst");
        for layer in 0..self.num_layers {
            let s = self.block_span(layer, src.index);
            let d = self.block_span(layer, dst.index);
            // Disjoint spans in one slice: split so both borrows can coexist.
            if s.start < d.start {
                let (lo, hi) = cache.split_at_mut(d.start);
                hi[..self.block_stride()].copy_from_slice(&lo[s]);
            } else {
                let (lo, hi) = cache.split_at_mut(s.start);
                lo[d].copy_from_slice(&hi[..self.block_stride()]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> KvLayout {
        KvLayout::new(3, 4, 2, 2, 2)
    }

    #[test]
    fn test_index_is_dense_and_unique() {
        let l = layout();
        let mut seen = vec![false; l.total_floats()];
        for layer in 0..l.num_layers {
            for b in 0..l.num_blocks {
                for t in 0..l.block_size {
                    for h in 0..l.num_kv_heads {
                        for v in [false, true] {
                            let i = l.index(layer, b, t, h, v);
                            for slot in seen.iter_mut().skip(i).take(l.head_dim) {
                                assert!(!*slot, "overlapping kv slot at {i}");
                                *slot = true;
                            }
                        }
                    }
                }
            }
        }
        assert!(seen.into_iter().all(|s| s), "kv layout leaves holes");
    }

    #[test]
    fn test_block_span_covers_every_entry_in_that_block() {
        let l = layout();
        for layer in 0..l.num_layers {
            for b in 0..l.num_blocks {
                let span = l.block_span(layer, b);
                for t in 0..l.block_size {
                    for h in 0..l.num_kv_heads {
                        for v in [false, true] {
                            let i = l.index(layer, b, t, h, v);
                            assert!(span.contains(&i) && span.contains(&(i + l.head_dim - 1)));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_copy_block_duplicates_all_layers_both_directions() {
        let l = layout();
        for (src, dst) in [(1_usize, 3_usize), (3, 1)] {
            let mut cache: Vec<f32> = (0..l.total_floats()).map(|i| i as f32).collect();
            l.copy_block(
                &mut cache,
                PhysicalBlock { index: src },
                PhysicalBlock { index: dst },
            );
            for layer in 0..l.num_layers {
                let s = l.block_span(layer, src);
                let d = l.block_span(layer, dst);
                assert_eq!(cache[s], cache[d], "layer {layer} not copied");
            }
            // Untouched blocks keep their original contents.
            for b in 0..l.num_blocks {
                if b == dst {
                    continue;
                }
                for layer in 0..l.num_layers {
                    let span = l.block_span(layer, b);
                    let start = span.start;
                    for (off, val) in cache[span].iter().enumerate() {
                        assert_eq!(*val, (start + off) as f32, "block {b} was clobbered");
                    }
                }
            }
        }
    }
}
