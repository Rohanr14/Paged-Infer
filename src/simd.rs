//! Vectorized inner loops for the matvec-bound decode path.
//!
//! Decoding one token is a sequence of matrix-vector products, and a matvec is
//! memory-bound: it streams the whole weight matrix to do one multiply-add per
//! element. The job of these kernels is to keep the FMA units fed while the
//! loads run, which means wide loads and enough independent accumulators to
//! cover FMA latency.
//!
//! Each kernel carries four accumulators. An FMA has ~4-cycle latency and
//! 0.5-cycle throughput, so a single accumulator serializes on its own
//! dependency chain and reaches about a quarter of peak; four independent
//! chains cover the latency.
//!
//! Dispatch is by target architecture, with a runtime feature check on x86_64
//! (AVX2 is not guaranteed) and none on aarch64 (NEON is in the baseline ABI).
//! Every path is checked against the scalar reference in `tests/simd_tests.rs`.

/// The obvious loop, with one accumulator.
///
/// Kept as the benchmark baseline because it is what the compiler cannot fix
/// for you. Rust does not permit reassociating f32 additions — they are not
/// associative — so LLVM must preserve this serial dependency chain and cannot
/// vectorize the reduction. That constraint is the entire reason the kernels
/// below are written by hand.
#[inline]
pub fn dot_naive(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0_f32;
    for i in 0..n {
        acc += a[i] * b[i];
    }
    acc
}

/// Portable reference. Every vectorized kernel must agree with this.
#[inline]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0.0_f32; 4];
    let mut i = 0;
    // Four accumulators even here: it lets the compiler vectorize the loop on
    // targets without a hand-written kernel.
    while i + 4 <= n {
        acc[0] += a[i] * b[i];
        acc[1] += a[i + 1] * b[i + 1];
        acc[2] += a[i + 2] * b[i + 2];
        acc[3] += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while i < n {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

/// Portable reference for the int8 weight path: `sum(w[i] as f32 * x[i])`.
#[inline]
pub fn dot_i8_scalar(w: &[i8], x: &[f32]) -> f32 {
    let n = w.len().min(x.len());
    let mut acc = [0.0_f32; 4];
    let mut i = 0;
    while i + 4 <= n {
        acc[0] += w[i] as f32 * x[i];
        acc[1] += w[i + 1] as f32 * x[i + 1];
        acc[2] += w[i + 2] as f32 * x[i + 2];
        acc[3] += w[i + 3] as f32 * x[i + 3];
        i += 4;
    }
    let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while i < n {
        total += w[i] as f32 * x[i];
        i += 1;
    }
    total
}

/// Portable reference for `out += weight * v`.
#[inline]
pub fn axpy_scalar(out: &mut [f32], weight: f32, v: &[f32]) {
    for (o, x) in out.iter_mut().zip(v.iter()) {
        *o += weight * x;
    }
}

// ── x86_64: AVX2 + FMA ───────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use std::arch::x86_64::*;

    /// Whether the AVX2+FMA kernels can run here. `is_x86_feature_detected!`
    /// caches its answer in a static, so this is a relaxed atomic load.
    #[inline]
    pub fn available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
        _mm_cvtss_f32(s)
    }

    /// # Safety
    /// Caller must have checked [`available`].
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut i = 0;
        while i + 32 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i + 24)),
                acc3,
            );
            i += 32;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }

        let sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut total = hsum(sum);
        while i < n {
            total += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        total
    }

    /// int8 weights against f32 activations.
    ///
    /// `_mm256_cvtepi8_epi32` sign-extends 8 bytes straight to 8 lanes of i32,
    /// so the widening costs one instruction per vector and the accumulation
    /// stays in f32 — no need to track a separate integer accumulator scale.
    ///
    /// # Safety
    /// Caller must have checked [`available`].
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_i8(w: &[i8], x: &[f32]) -> f32 {
        let n = w.len().min(x.len());
        let pw = w.as_ptr();
        let px = x.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        #[inline(always)]
        unsafe fn widen(p: *const i8) -> __m256 {
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p as *const __m128i)))
        }

        let mut i = 0;
        while i + 32 <= n {
            acc0 = _mm256_fmadd_ps(widen(pw.add(i)), _mm256_loadu_ps(px.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(widen(pw.add(i + 8)), _mm256_loadu_ps(px.add(i + 8)), acc1);
            acc2 = _mm256_fmadd_ps(widen(pw.add(i + 16)), _mm256_loadu_ps(px.add(i + 16)), acc2);
            acc3 = _mm256_fmadd_ps(widen(pw.add(i + 24)), _mm256_loadu_ps(px.add(i + 24)), acc3);
            i += 32;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(widen(pw.add(i)), _mm256_loadu_ps(px.add(i)), acc0);
            i += 8;
        }

        let sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut total = hsum(sum);
        while i < n {
            total += *pw.add(i) as f32 * *px.add(i);
            i += 1;
        }
        total
    }

    /// # Safety
    /// Caller must have checked [`available`].
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy(out: &mut [f32], weight: f32, v: &[f32]) {
        let n = out.len().min(v.len());
        let po = out.as_mut_ptr();
        let pv = v.as_ptr();
        let w = _mm256_set1_ps(weight);
        let mut i = 0;
        while i + 8 <= n {
            let acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(pv.add(i)), _mm256_loadu_ps(po.add(i)));
            _mm256_storeu_ps(po.add(i), acc);
            i += 8;
        }
        while i < n {
            *po.add(i) += weight * *pv.add(i);
            i += 1;
        }
    }
}

// ── aarch64: NEON (Apple Silicon, Graviton) ──────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use std::arch::aarch64::*;

    /// NEON is mandatory in the aarch64 baseline, so there is nothing to probe.
    #[inline]
    pub fn available() -> bool {
        true
    }

    /// # Safety
    /// Safe on any aarch64 target; `unsafe` only because the intrinsics are.
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0;
        while i + 16 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            i += 4;
        }

        let sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut total = vaddvq_f32(sum);
        while i < n {
            total += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        total
    }

    /// # Safety
    /// Safe on any aarch64 target; `unsafe` only because the intrinsics are.
    pub unsafe fn dot_i8(w: &[i8], x: &[f32]) -> f32 {
        let n = w.len().min(x.len());
        let pw = w.as_ptr();
        let px = x.as_ptr();
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        // i8x8 -> i16x8 -> two i32x4 -> two f32x4.
        #[inline(always)]
        unsafe fn widen8(p: *const i8) -> (float32x4_t, float32x4_t) {
            let wide = vmovl_s8(vld1_s8(p));
            (
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide))),
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide))),
            )
        }

        let mut i = 0;
        while i + 16 <= n {
            let (lo, hi) = widen8(pw.add(i));
            acc0 = vfmaq_f32(acc0, lo, vld1q_f32(px.add(i)));
            acc1 = vfmaq_f32(acc1, hi, vld1q_f32(px.add(i + 4)));
            let (lo2, hi2) = widen8(pw.add(i + 8));
            acc2 = vfmaq_f32(acc2, lo2, vld1q_f32(px.add(i + 8)));
            acc3 = vfmaq_f32(acc3, hi2, vld1q_f32(px.add(i + 12)));
            i += 16;
        }
        while i + 8 <= n {
            let (lo, hi) = widen8(pw.add(i));
            acc0 = vfmaq_f32(acc0, lo, vld1q_f32(px.add(i)));
            acc1 = vfmaq_f32(acc1, hi, vld1q_f32(px.add(i + 4)));
            i += 8;
        }

        let sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut total = vaddvq_f32(sum);
        while i < n {
            total += *pw.add(i) as f32 * *px.add(i);
            i += 1;
        }
        total
    }

    /// # Safety
    /// Safe on any aarch64 target; `unsafe` only because the intrinsics are.
    pub unsafe fn axpy(out: &mut [f32], weight: f32, v: &[f32]) {
        let n = out.len().min(v.len());
        let po = out.as_mut_ptr();
        let pv = v.as_ptr();
        let w = vdupq_n_f32(weight);
        let mut i = 0;
        while i + 4 <= n {
            vst1q_f32(
                po.add(i),
                vfmaq_f32(vld1q_f32(po.add(i)), w, vld1q_f32(pv.add(i))),
            );
            i += 4;
        }
        while i < n {
            *po.add(i) += weight * *pv.add(i);
            i += 1;
        }
    }
}

// ── dispatch ─────────────────────────────────────────────────────────────────

/// Name of the kernel family that will actually run here. Reported by the
/// benchmarks so a number is never attributed to the wrong code path.
pub fn backend() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::available() {
            return "avx2+fma";
        }
        "scalar (no avx2/fma)"
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "scalar (portable)"
    }
}

/// True when a hand-written vector kernel is in use.
pub fn vectorized() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        x86::available()
    }
    #[cfg(target_arch = "aarch64")]
    {
        return true;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::available() {
            return unsafe { x86::dot(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::dot(a, b) };
    }
    #[allow(unreachable_code)]
    dot_scalar(a, b)
}

#[inline]
pub fn dot_i8(w: &[i8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::available() {
            return unsafe { x86::dot_i8(w, x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::dot_i8(w, x) };
    }
    #[allow(unreachable_code)]
    dot_i8_scalar(w, x)
}

#[inline]
pub fn axpy(out: &mut [f32], weight: f32, v: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::available() {
            unsafe { x86::axpy(out, weight, v) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::axpy(out, weight, v) };
        return;
    }
    #[allow(unreachable_code)]
    axpy_scalar(out, weight, v)
}
