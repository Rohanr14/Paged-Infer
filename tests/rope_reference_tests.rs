//! Analytic checks for Llama 3's frequency transition, independent of the
//! production wavelength branches. The checkpoint/Transformers fixtures check
//! compatibility; these tests isolate the interpolation and its two cutoffs.

use paged_infer::math::{rope_inv_freq, RopeScaling};

const ORIGINAL_CONTEXT: usize = 8192;
const LOW_CYCLES: f64 = 1.0;
const HIGH_CYCLES: f64 = 4.0;

fn scaling(factor: f32) -> RopeScaling {
    RopeScaling::Llama3 {
        factor,
        low_freq_factor: LOW_CYCLES as f32,
        high_freq_factor: HIGH_CYCLES as f32,
        original_max_position_embeddings: ORIGINAL_CONTEXT,
    }
}

/// Express the scaling as a clipped linear blend in *cycles per original
/// context*, using f64 throughout. This avoids both the f32 arithmetic and the
/// wavelength-based branch structure of `rope_inv_freq`.
fn analytic_frequency(head_dim: usize, theta: f32, pair: usize, factor: f32) -> f64 {
    let base = f64::from(theta).powf(-2.0 * pair as f64 / head_dim as f64);
    let cycles = ORIGINAL_CONTEXT as f64 * base / std::f64::consts::TAU;
    let unscaled_share = ((cycles - LOW_CYCLES) / (HIGH_CYCLES - LOW_CYCLES)).clamp(0.0, 1.0);
    base * (1.0 / f64::from(factor) + unscaled_share * (1.0 - 1.0 / f64::from(factor)))
}

fn assert_frequency(actual: f32, expected: f64, context: &str) {
    // f32 powf, reciprocal and interpolation each round; this relative bound
    // covers those errors without letting the tiny slow frequencies disappear
    // beneath a fixed absolute epsilon. A missing or reversed blend fails it.
    let relative_error = (f64::from(actual) / expected - 1.0).abs();
    assert!(
        actual.is_finite() && relative_error <= 8.0 * f64::from(f32::EPSILON),
        "{context}: {actual:e} vs {expected:e}, relative error {relative_error:e}"
    );
}

#[test]
fn llama3_frequencies_match_an_independent_f64_blend_in_every_band() {
    let theta = 500_000.0;
    for head_dim in [64, 128] {
        for factor in [8.0, 32.0] {
            let frequencies = rope_inv_freq(head_dim, theta, scaling(factor));
            let mut bands = [0; 3];
            for (pair, &actual) in frequencies.iter().enumerate() {
                let base = f64::from(theta).powf(-2.0 * pair as f64 / head_dim as f64);
                let cycles = ORIGINAL_CONTEXT as f64 * base / std::f64::consts::TAU;
                let band = if cycles < LOW_CYCLES {
                    0
                } else if cycles > HIGH_CYCLES {
                    2
                } else {
                    1
                };
                bands[band] += 1;
                assert_frequency(
                    actual,
                    analytic_frequency(head_dim, theta, pair, factor),
                    &format!("head_dim {head_dim}, factor {factor}, pair {pair}, band {band}"),
                );
            }
            assert!(
                bands.iter().all(|&count| count > 0),
                "must exercise slow, interpolated and unchanged frequencies: {bands:?}"
            );
        }
    }
}

#[test]
fn llama3_blend_is_correct_on_both_sides_of_each_cutoff() {
    // A four-dimensional head has a second frequency theta^(-1/2). Choose
    // theta so that this pair lands at and near both cutoffs, as well as at
    // 25%, 50% and 75% of the ramp. Real checkpoints need not contain a pair
    // this close to either cutoff, so fixture-only coverage can miss defects.
    let cycle_counts = [
        0.5,
        LOW_CYCLES - 1e-4,
        LOW_CYCLES,
        LOW_CYCLES + 1e-4,
        1.75,
        2.5,
        3.25,
        HIGH_CYCLES - 1e-4,
        HIGH_CYCLES,
        HIGH_CYCLES + 1e-4,
        4.5,
    ];
    for factor in [8.0, 32.0] {
        for cycles in cycle_counts {
            let base = cycles * std::f64::consts::TAU / ORIGINAL_CONTEXT as f64;
            let theta = base.powi(-2) as f32;
            let actual = rope_inv_freq(4, theta, scaling(factor))[1];
            // Use the representable theta actually supplied to Rust: rounding
            // the constructed theta can move an exact cutoff slightly.
            assert_frequency(
                actual,
                analytic_frequency(4, theta, 1, factor),
                &format!("factor {factor}, target cycles {cycles}"),
            );
        }
    }
}
