use paged_infer::math::paged_attention;
use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;

#[test]
fn test_block_table_fragmented_mapping_roundtrip() {
    let mut allocator = BlockAllocator::new(8, 4);
    let mut bt = BlockTable::new();

    let b0 = allocator.allocate().unwrap();
    let b1 = allocator.allocate().unwrap();
    allocator.free(b0);
    let b2 = allocator.allocate().unwrap();

    bt.append_block(b1);
    bt.append_block(b2);

    let (p0, off0) = bt.get_physical_location(0, 4).unwrap();
    let (p5, off5) = bt.get_physical_location(5, 4).unwrap();

    assert_eq!(p0.index, b1.index);
    assert_eq!(off0, 0);
    assert_eq!(p5.index, b2.index);
    assert_eq!(off5, 1);
}

#[test]
fn test_paged_attention_prefers_closer_key() {
    let q = [1.0_f32, 0.0];
    let k0 = [1.0_f32, 0.0];
    let k1 = [0.0_f32, 1.0];
    let v0 = [2.0_f32, 0.0];
    let v1 = [0.0_f32, 4.0];

    let mut scores = [0.0_f32; 2];
    let mut out = [0.0_f32; 2];
    paged_attention(&q, &[&k0, &k1], &[&v0, &v1], &mut scores, &mut out);

    assert!(scores[0] > scores[1]);
    assert!(out[0] > out[1]);
}
