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

    /// One weight row against `BT` activation vectors, loading `w` once.
    ///
    /// Accumulator structure is identical to [`dot`]: chunk `j` of every
    /// 32-element group lands in `acc[b][j]`, the 8-element tail lands in
    /// `acc[b][0]`, the reduction is `(0+1)+(2+3)`, and the scalar remainder is
    /// added last in increasing index. `BT == 1` therefore reproduces [`dot`]
    /// bit for bit.
    ///
    /// `BT` is bounded by the register file, not by taste: `1 + 4*BT` live
    /// vectors must fit in 16 ymm registers or the accumulators spill and the
    /// saved loads come straight back as stores.
    ///
    /// # Safety
    /// Caller must have checked [`available`], and `x_block` must hold at least
    /// `BT * stride` elements.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_multi<const BT: usize>(
        w: &[f32],
        x_block: &[f32],
        stride: usize,
    ) -> [f32; BT] {
        let n = w.len().min(stride);
        let pw = w.as_ptr();
        let px = x_block.as_ptr();
        let mut acc = [[_mm256_setzero_ps(); 4]; BT];

        let mut i = 0;
        while i + 32 <= n {
            for (j, off) in [0usize, 8, 16, 24].into_iter().enumerate() {
                let wv = _mm256_loadu_ps(pw.add(i + off));
                for (b, acc_b) in acc.iter_mut().enumerate() {
                    acc_b[j] = _mm256_fmadd_ps(
                        wv,
                        _mm256_loadu_ps(px.add(b * stride + i + off)),
                        acc_b[j],
                    );
                }
            }
            i += 32;
        }
        while i + 8 <= n {
            let wv = _mm256_loadu_ps(pw.add(i));
            for (b, acc_b) in acc.iter_mut().enumerate() {
                acc_b[0] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(px.add(b * stride + i)), acc_b[0]);
            }
            i += 8;
        }

        std::array::from_fn(|b| {
            let sum = _mm256_add_ps(
                _mm256_add_ps(acc[b][0], acc[b][1]),
                _mm256_add_ps(acc[b][2], acc[b][3]),
            );
            let mut total = hsum(sum);
            let mut k = i;
            while k < n {
                total += *pw.add(k) * *px.add(b * stride + k);
                k += 1;
            }
            total
        })
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

    /// One int8 weight row against `BT` activation vectors.
    ///
    /// The saving here is larger than in the f32 case. Every weight element has
    /// to be widened from `i8` to `f32` before it can be multiplied — a load, a
    /// sign-extend and a convert — and the naive nesting redoes all three once
    /// per batch entry. Widening once and reusing the vector across the tile
    /// removes `BT-1` copies of that work, not just `BT-1` loads.
    ///
    /// Accumulator structure matches [`dot_i8`] exactly, so `BT == 1`
    /// reproduces it bit for bit.
    ///
    /// # Safety
    /// Caller must have checked [`available`]; `x_block` must hold at least
    /// `BT * stride` elements.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_i8_multi<const BT: usize>(
        w: &[i8],
        x_block: &[f32],
        stride: usize,
    ) -> [f32; BT] {
        let n = w.len().min(stride);
        let pw = w.as_ptr();
        let px = x_block.as_ptr();
        let mut acc = [[_mm256_setzero_ps(); 4]; BT];

        #[inline(always)]
        unsafe fn widen(p: *const i8) -> __m256 {
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p as *const __m128i)))
        }

        let mut i = 0;
        while i + 32 <= n {
            for (j, off) in [0usize, 8, 16, 24].into_iter().enumerate() {
                let wv = widen(pw.add(i + off));
                for (b, acc_b) in acc.iter_mut().enumerate() {
                    acc_b[j] = _mm256_fmadd_ps(
                        wv,
                        _mm256_loadu_ps(px.add(b * stride + i + off)),
                        acc_b[j],
                    );
                }
            }
            i += 32;
        }
        while i + 8 <= n {
            let wv = widen(pw.add(i));
            for (b, acc_b) in acc.iter_mut().enumerate() {
                acc_b[0] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(px.add(b * stride + i)), acc_b[0]);
            }
            i += 8;
        }

        std::array::from_fn(|b| {
            let sum = _mm256_add_ps(
                _mm256_add_ps(acc[b][0], acc[b][1]),
                _mm256_add_ps(acc[b][2], acc[b][3]),
            );
            let mut total = hsum(sum);
            let mut k = i;
            while k < n {
                total += *pw.add(k) as f32 * *px.add(b * stride + k);
                k += 1;
            }
            total
        })
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

    /// One weight row against `BT` activation vectors, loading `w` once.
    /// Mirrors [`dot`]'s accumulator structure exactly, so `BT == 1`
    /// reproduces it bit for bit. aarch64 has 32 vector registers, so `BT` can
    /// go wider here than on AVX2 before `1 + 4*BT` accumulators spill.
    ///
    /// # Safety
    /// Safe on any aarch64 target; `x_block` must hold at least `BT * stride`
    /// elements.
    pub unsafe fn dot_multi<const BT: usize>(
        w: &[f32],
        x_block: &[f32],
        stride: usize,
    ) -> [f32; BT] {
        let n = w.len().min(stride);
        let pw = w.as_ptr();
        let px = x_block.as_ptr();
        let mut acc = [[vdupq_n_f32(0.0); 4]; BT];

        let mut i = 0;
        while i + 16 <= n {
            for j in 0..4 {
                let wv = vld1q_f32(pw.add(i + 4 * j));
                for b in 0..BT {
                    acc[b][j] = vfmaq_f32(acc[b][j], wv, vld1q_f32(px.add(b * stride + i + 4 * j)));
                }
            }
            i += 16;
        }
        while i + 4 <= n {
            let wv = vld1q_f32(pw.add(i));
            for (b, acc_b) in acc.iter_mut().enumerate() {
                acc_b[0] = vfmaq_f32(acc_b[0], wv, vld1q_f32(px.add(b * stride + i)));
            }
            i += 4;
        }

        std::array::from_fn(|b| {
            let sum = vaddq_f32(
                vaddq_f32(acc[b][0], acc[b][1]),
                vaddq_f32(acc[b][2], acc[b][3]),
            );
            let mut total = vaddvq_f32(sum);
            let mut k = i;
            while k < n {
                total += *pw.add(k) * *px.add(b * stride + k);
                k += 1;
            }
            total
        })
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

    /// One int8 weight row against `BT` activation vectors, widening each
    /// weight element once instead of once per batch entry. Mirrors
    /// [`dot_i8`]'s accumulator structure, so `BT == 1` reproduces it exactly.
    ///
    /// # Safety
    /// Safe on any aarch64 target; `x_block` must hold at least `BT * stride`
    /// elements.
    pub unsafe fn dot_i8_multi<const BT: usize>(
        w: &[i8],
        x_block: &[f32],
        stride: usize,
    ) -> [f32; BT] {
        let n = w.len().min(stride);
        let pw = w.as_ptr();
        let px = x_block.as_ptr();
        let mut acc = [[vdupq_n_f32(0.0); 4]; BT];

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
            let (lo2, hi2) = widen8(pw.add(i + 8));
            for (b, acc_b) in acc.iter_mut().enumerate() {
                let xb = px.add(b * stride);
                acc_b[0] = vfmaq_f32(acc_b[0], lo, vld1q_f32(xb.add(i)));
                acc_b[1] = vfmaq_f32(acc_b[1], hi, vld1q_f32(xb.add(i + 4)));
                acc_b[2] = vfmaq_f32(acc_b[2], lo2, vld1q_f32(xb.add(i + 8)));
                acc_b[3] = vfmaq_f32(acc_b[3], hi2, vld1q_f32(xb.add(i + 12)));
            }
            i += 16;
        }
        while i + 8 <= n {
            let (lo, hi) = widen8(pw.add(i));
            for (b, acc_b) in acc.iter_mut().enumerate() {
                let xb = px.add(b * stride);
                acc_b[0] = vfmaq_f32(acc_b[0], lo, vld1q_f32(xb.add(i)));
                acc_b[1] = vfmaq_f32(acc_b[1], hi, vld1q_f32(xb.add(i + 4)));
            }
            i += 8;
        }

        std::array::from_fn(|b| {
            let sum = vaddq_f32(
                vaddq_f32(acc[b][0], acc[b][1]),
                vaddq_f32(acc[b][2], acc[b][3]),
            );
            let mut total = vaddvq_f32(sum);
            let mut k = i;
            while k < n {
                total += *pw.add(k) as f32 * *px.add(b * stride + k);
                k += 1;
            }
            total
        })
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

// ── one weight row against several activation vectors ────────────────────────
//
// The batched matmul's inner loop is `for b { dot(w_row, x_b) }`. That reloads
// every element of `w_row` once per batch entry: at batch 8, eight passes over
// the same row, each doing one load of `w` and one load of `x` per fused
// multiply-add. Two loads to feed one FMA is the wrong ratio on every machine
// measured — load ports retire two per cycle and FMAs also two per cycle, so
// the loads take twice as long as the arithmetic they feed and the kernel is
// issue-bound rather than bandwidth-bound. That is what stops batched decode
// scaling past batch four or five, not anything to do with attention.
//
// `dot_multi` loads each element of `w` **once** and multiplies it into `BT`
// independent accumulator sets, cutting the ratio from `2` loads per FMA
// toward `1 + 1/BT`.
//
// The accumulator layout is copied from `dot` rather than simplified, on
// purpose. Four accumulators per output, the same 32- (or 16-) element
// chunking, the same `(acc0+acc1)+(acc2+acc3)` reduction, the same scalar tail
// order — so `dot_multi::<1>` is `dot`, element for element, and the tiled
// batched kernel stays bit-identical to the single-sequence one. A tile that
// collapsed to one accumulator per output would be faster still and would
// quietly break that.

/// `out[j] = dot(w, &x_block[j * stride..][..stride])`, for `j` in `0..BT`.
///
/// `x_block` holds `BT` consecutive activation vectors of length `stride`.
pub fn dot_multi<const BT: usize>(w: &[f32], x_block: &[f32], stride: usize) -> [f32; BT] {
    debug_assert!(x_block.len() >= BT * stride);
    #[cfg(target_arch = "x86_64")]
    if x86::available() {
        return unsafe { x86::dot_multi::<BT>(w, x_block, stride) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::dot_multi::<BT>(w, x_block, stride) };
    }
    #[allow(unreachable_code)]
    dot_multi_scalar::<BT>(w, x_block, stride)
}

/// Portable reference: literally `BT` calls to [`dot_scalar`].
pub fn dot_multi_scalar<const BT: usize>(w: &[f32], x_block: &[f32], stride: usize) -> [f32; BT] {
    std::array::from_fn(|j| dot_scalar(w, &x_block[j * stride..(j + 1) * stride]))
}

/// `out[j] = dot_i8(w, &x_block[j * stride..][..stride])`, for `j` in `0..BT`.
pub fn dot_i8_multi<const BT: usize>(w: &[i8], x_block: &[f32], stride: usize) -> [f32; BT] {
    debug_assert!(x_block.len() >= BT * stride);
    #[cfg(target_arch = "x86_64")]
    if x86::available() {
        return unsafe { x86::dot_i8_multi::<BT>(w, x_block, stride) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::dot_i8_multi::<BT>(w, x_block, stride) };
    }
    #[allow(unreachable_code)]
    std::array::from_fn(|j| dot_i8_scalar(w, &x_block[j * stride..(j + 1) * stride]))
}
