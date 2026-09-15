//! Shared-value AXPY must only change the schedule: each output row retains
//! exactly the arithmetic of a separate AXPY, including each backend's tail.

use paged_infer::simd;

fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let bits = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            ((bits % 4001) as f32 - 2000.0) / 997.0
        })
        .collect()
}

fn check_rows<const BT: usize>(scalar: bool) {
    // Sweep all vector and scalar tails, with both packed and padded rows.
    for n in (0..80).chain([96, 127, 128, 129, 257]) {
        for padding in [0, 1, 7] {
            let stride = n + padding;
            // No trailing padding after the last row is required. Extra
            // prefix/suffix sentinels also catch writes outside the slice.
            let required = (BT - 1) * stride + n;
            let mut expected = values(required + 6, 71);
            let mut actual = expected.clone();
            for pass in 0..7 {
                let v = values(n, 101 + pass);
                let weights = std::array::from_fn(|b| {
                    // Include zero and both signs; the non-dyadic values
                    // exercise FMA rounding differences on vector lanes.
                    [0.0, -1.0000001, 0.33333334, 13.12345][(b + pass as usize) % 4]
                });
                for (b, &weight) in weights.iter().enumerate() {
                    let row = &mut expected[3 + b * stride..3 + b * stride + n];
                    if scalar {
                        simd::axpy_scalar(row, weight, &v);
                    } else {
                        simd::axpy(row, weight, &v);
                    }
                }
                if scalar {
                    simd::axpy_multi_scalar::<BT>(
                        &mut actual[3..3 + required],
                        weights,
                        &v,
                        stride,
                    );
                } else {
                    simd::axpy_multi::<BT>(&mut actual[3..3 + required], weights, &v, stride);
                }
                for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
                    assert!(got.is_finite() && want.is_finite());
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "BT={BT}, n={n}, padding={padding}, pass={pass}, index={i}, scalar={scalar}"
                    );
                }
            }
        }
    }
}

#[test]
fn multi_output_matches_per_row_axpy_exactly() {
    check_rows::<1>(false);
    check_rows::<2>(false);
    check_rows::<3>(false);
    check_rows::<4>(false);
    check_rows::<6>(false);
}

#[test]
fn scalar_multi_output_matches_per_row_scalar_exactly() {
    check_rows::<1>(true);
    check_rows::<2>(true);
    check_rows::<3>(true);
    check_rows::<4>(true);
    check_rows::<6>(true);
}

#[test]
fn multi_output_zero_rows_are_a_noop() {
    let mut out = [1.5, -2.25];
    simd::axpy_multi::<0>(&mut out, [], &[1.0, 2.0, 3.0], usize::MAX);
    simd::axpy_multi_scalar::<0>(&mut out, [], &[1.0, 2.0, 3.0], 0);
    assert_eq!(out, [1.5, -2.25]);
}

#[test]
fn multi_output_rejects_invalid_layouts_before_writing() {
    for scalar in [false, true] {
        for (out_len, n, stride) in [
            (12, 4, 3),             // Overlapping rows even though the slice has room.
            (18, 3, 8),             // Final row is one element short.
            (0, 0, usize::MAX),     // Multiplication overflow for three rows.
            (0, 2, usize::MAX / 2), // Final-row addition overflow.
        ] {
            let mut out = vec![123.0; out_len];
            let v = vec![1.0; n];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if scalar {
                    simd::axpy_multi_scalar::<3>(&mut out, [1.0; 3], &v, stride);
                } else {
                    simd::axpy_multi::<3>(&mut out, [1.0; 3], &v, stride);
                }
            }));
            assert!(result.is_err(), "layout unexpectedly accepted");
            assert!(out.iter().all(|&value| value == 123.0));
        }
    }
}

fn check_weighted_block<const BT: usize>() {
    for dim in [0, 1, 3, 4, 5, 7, 8, 9, 31, 32, 33, 64] {
        for tokens in [0, 1, 3, 16, 31] {
            let stride = dim + 3;
            let value_stride = dim + 5;
            let len = if BT == 0 { 0 } else { (BT - 1) * stride + dim };
            let mut actual: Vec<f32> = (0..len + 2).map(|i| i as f32 * 0.013 - 0.4).collect();
            let mut expected = actual.clone();
            let mut values: Vec<f32> = (0..tokens * value_stride)
                .map(|i| i as f32 * 0.017 - 1.1)
                .collect();
            let weights: [Vec<f32>; BT] = std::array::from_fn(|b| {
                (0..tokens)
                    .map(|t| {
                        if t % 7 == 0 {
                            0.0
                        } else {
                            (b * 13 + t) as f32 * 0.021 - 0.3
                        }
                    })
                    .collect()
            });
            // A value with zero weight for all outputs must not contaminate them.
            for t in (0..tokens).step_by(7) {
                values[t * value_stride..][..dim].fill(f32::NAN);
            }
            for t in 0..tokens {
                for b in 0..BT {
                    if weights[b][t] != 0.0 {
                        simd::axpy(
                            &mut expected[1 + b * stride..][..dim],
                            weights[b][t],
                            &values[t * value_stride..][..dim],
                        );
                    }
                }
            }
            simd::weighted_sum_multi::<BT>(
                &mut actual[1..1 + len],
                stride,
                std::array::from_fn(|b| weights[b].as_slice()),
                &values,
                value_stride,
                dim,
            );
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "BT={BT} dim={dim} tokens={tokens}; output guards and padding must survive"
            );
        }
    }
}

#[test]
fn block_value_sums_match_tokenwise_axpy_with_masks_and_simd_tails() {
    check_weighted_block::<0>();
    check_weighted_block::<1>();
    check_weighted_block::<2>();
    check_weighted_block::<3>();
    check_weighted_block::<4>();
}

fn check_independent_weight_rows<const BT: usize>(weights: [&[f32]; BT]) {
    let tokens = weights[0].len();
    for dim in [3, 4, 7, 8, 9, 17] {
        let stride = dim + 3;
        let value_stride = dim + 5;
        let len = (BT - 1) * stride + dim;
        let mut actual = values(len + 2, 29);
        for (b, row) in weights.iter().enumerate() {
            if row.iter().all(|&weight| weight == 0.0) {
                // Skipped rows must retain even the sign of their zeros.
                actual[1 + b * stride..][..dim].fill(-0.0);
            }
        }
        let mut expected = actual.clone();
        // Omit final-row padding and poison the inter-row padding so using
        // the wrong value stride cannot accidentally produce a valid sum.
        let mut block = vec![f32::NAN; (tokens - 1) * value_stride + dim];
        for t in 0..tokens {
            block[t * value_stride..][..dim].copy_from_slice(&values(dim, 71 + t as u32));
            for (b, row) in weights.iter().enumerate() {
                if row[t] != 0.0 {
                    simd::axpy(
                        &mut expected[1 + b * stride..][..dim],
                        row[t],
                        &block[t * value_stride..][..dim],
                    );
                }
            }
        }
        simd::weighted_sum_multi::<BT>(
            &mut actual[1..1 + len],
            stride,
            weights,
            &block,
            value_stride,
            dim,
        );
        assert!(actual.iter().all(|value| value.is_finite()));
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "BT={BT}, dim={dim}, tokens={tokens}; independent masks, tails and padding must match"
        );
    }
}

#[test]
fn single_token_value_sums_cover_nonzero_and_mixed_zero_paths() {
    for weights in [
        [0.33333334, -1.0000001, 13.12345, -0.375],
        [0.0, 0.33333334, -0.0, -1.0000001],
    ] {
        let rows = weights.map(|weight| [weight]);
        check_independent_weight_rows::<1>(std::array::from_fn(|b| rows[b].as_slice()));
        check_independent_weight_rows::<2>(std::array::from_fn(|b| rows[b].as_slice()));
        check_independent_weight_rows::<3>(std::array::from_fn(|b| rows[b].as_slice()));
        check_independent_weight_rows::<4>(std::array::from_fn(|b| rows[b].as_slice()));
    }
}

#[test]
fn block_value_sums_apply_zero_masks_independently_to_each_row() {
    // Every token contributes to some rows and is masked in others. In
    // particular, a zero in row 0 must not suppress the other rows' updates.
    let weights = [
        [0.0, 0.33333334, -0.0, 0.25, 0.0, -0.33333334],
        [0.25, 0.0, -0.5, 0.0, 0.75, -0.0],
        [-0.5, 1.0000001, 0.0, -0.33333334, -0.0, 0.375],
        [-0.0, -1.0000001, 0.33333334, 0.0, 0.17, 0.0],
    ];
    check_independent_weight_rows::<2>(std::array::from_fn(|b| weights[b].as_slice()));
    check_independent_weight_rows::<3>(std::array::from_fn(|b| weights[b].as_slice()));
    check_independent_weight_rows::<4>(std::array::from_fn(|b| weights[b].as_slice()));
}

#[test]
fn block_value_sums_reject_invalid_layouts_before_writing() {
    for case in 0..4 {
        let mut out = vec![17.0; 8];
        let weights = [1.0; 3];
        let values = [2.0; 32];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match case {
            0 => {
                simd::weighted_sum_multi::<2>(&mut out, 4, [&weights, &weights[..2]], &values, 8, 4)
            }
            1 => simd::weighted_sum_multi::<2>(&mut out, 4, [&weights; 2], &values, 3, 4),
            2 => simd::weighted_sum_multi::<2>(&mut out, 4, [&weights; 2], &values[..19], 8, 4),
            _ => simd::weighted_sum_multi::<2>(
                &mut out,
                4,
                [&weights[..2]; 2],
                &values,
                usize::MAX,
                4,
            ),
        }));
        assert!(result.is_err());
        assert_eq!(out, vec![17.0; 8]);
    }
}
