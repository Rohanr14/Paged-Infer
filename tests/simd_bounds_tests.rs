//! Safe multi-dot APIs must validate layouts before dispatching to unsafe SIMD.
use paged_infer::simd;

#[test]
fn multi_dot_rejects_short_or_overflowing_layouts_in_release() {
    for (len, stride) in [(0, 8), (15, 8), (0, usize::MAX)] {
        let activations = vec![1.0; len];
        assert!(std::panic::catch_unwind(|| {
            simd::dot_multi::<2>(&[1.0; 8], &activations, stride)
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| {
            simd::dot_i8_multi::<2>(&[1; 8], &activations, stride)
        })
        .is_err());
    }
    assert_eq!(
        simd::dot_multi::<0>(&[1.0; 8], &[], usize::MAX),
        [0.0f32; 0]
    );
    assert_eq!(
        simd::dot_i8_multi::<0>(&[1; 8], &[], usize::MAX),
        [0.0f32; 0]
    );
}
