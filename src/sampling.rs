//! Token sampling.
//!
//! Greedy argmax is only correct when you want the single most likely
//! continuation. Anything that draws several samples from one prompt — which is
//! exactly what copy-on-write forking exists to make cheap — needs actual
//! sampling, or every branch returns the same text.
//!
//! Deterministic by construction: the RNG is a seeded xorshift, so a given
//! (seed, prompt) always replays identically. That matters more for an engine
//! than raw randomness quality, because it makes generation bugs reproducible.

/// xorshift64*, chosen for being a few lines and dependency-free. Not
/// cryptographic, and not meant to be.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            // A zero state is a fixed point for xorshift, so fold it away.
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits give exactly the f32 mantissa width.
        ((self.next_u64() >> 40) as f32) / ((1_u32 << 24) as f32)
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    /// `0.0` means greedy argmax.
    pub temperature: f32,
    /// Nucleus threshold; `1.0` disables truncation.
    pub top_p: f32,
    /// Keep only the `top_k` highest-probability tokens; `0` disables.
    pub top_k: usize,
    rng: Rng,
}

impl Sampler {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            rng: Rng::new(0),
        }
    }

    pub fn new(temperature: f32, top_p: f32, top_k: usize, seed: u64) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            rng: Rng::new(seed),
        }
    }

    /// Pick the next token. `logits` is consumed as scratch.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        assert!(!logits.is_empty(), "cannot sample from an empty distribution");

        if self.temperature <= 0.0 {
            return argmax(logits) as u32;
        }

        // Rank once; both top-k and top-p are prefix conditions on this order.
        let mut order: Vec<u32> = (0..logits.len() as u32).collect();
        order.sort_unstable_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));

        let keep = if self.top_k == 0 {
            order.len()
        } else {
            self.top_k.min(order.len())
        };
        order.truncate(keep);

        let inv_temp = 1.0 / self.temperature;
        let max_logit = logits[order[0] as usize];
        let mut probs: Vec<f32> = order
            .iter()
            .map(|&i| ((logits[i as usize] - max_logit) * inv_temp).exp())
            .collect();
        let total: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= total;
        }

        // Nucleus: smallest prefix whose mass reaches top_p. Always keeps at
        // least one token, so a peaked distribution still samples.
        if self.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut cut = probs.len();
            for (i, p) in probs.iter().enumerate() {
                cumulative += *p;
                if cumulative >= self.top_p {
                    cut = i + 1;
                    break;
                }
            }
            probs.truncate(cut);
            order.truncate(cut);
            let renorm: f32 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= renorm;
            }
        }

        let target = self.rng.next_f32();
        let mut cumulative = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumulative += *p;
            if target < cumulative {
                return order[i];
            }
        }
        // Falls through only on rounding; the last kept token is the right home
        // for the leftover mass.
        *order.last().expect("nucleus keeps at least one token")
    }
}

pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_temperature_is_greedy() {
        let mut s = Sampler::greedy();
        let mut logits = vec![0.1, 5.0, 0.3, 4.9];
        assert_eq!(s.sample(&mut logits), 1);
    }

    #[test]
    fn test_top_k_one_is_greedy_at_any_temperature() {
        let mut s = Sampler::new(2.0, 1.0, 1, 7);
        for _ in 0..50 {
            let mut logits = vec![0.1, 5.0, 0.3, 4.9];
            assert_eq!(s.sample(&mut logits), 1);
        }
    }

    #[test]
    fn test_sampling_is_deterministic_for_a_seed() {
        let draw = |seed| {
            let mut s = Sampler::new(1.0, 1.0, 0, seed);
            (0..32)
                .map(|_| s.sample(&mut vec![1.0, 2.0, 3.0, 0.5]))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(11), draw(11));
        assert_ne!(draw(11), draw(12), "different seeds should diverge");
    }

    #[test]
    fn test_sampling_explores_beyond_the_argmax() {
        // The point of forking n samples: they must not all be identical.
        let mut s = Sampler::new(1.0, 1.0, 0, 3);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(s.sample(&mut vec![1.0, 1.0, 1.0, 1.0]));
        }
        assert!(seen.len() > 1, "uniform logits should reach several tokens");
    }

    #[test]
    fn test_top_p_excludes_the_tail() {
        // Token 0 alone carries >90% of the mass, so nucleus 0.9 must never
        // reach the others.
        let mut s = Sampler::new(1.0, 0.9, 0, 5);
        for _ in 0..200 {
            assert_eq!(s.sample(&mut vec![10.0, 0.0, 0.0, 0.0]), 0);
        }
    }

    #[test]
    fn test_empirical_frequencies_track_the_distribution() {
        // Logits ln(3), ln(1) -> a 3:1 split.
        let mut s = Sampler::new(1.0, 1.0, 0, 99);
        let mut hits = [0_u32; 2];
        for _ in 0..20_000 {
            hits[s.sample(&mut vec![3.0_f32.ln(), 1.0_f32.ln()]) as usize] += 1;
        }
        let ratio = hits[0] as f64 / (hits[0] + hits[1]) as f64;
        assert!((ratio - 0.75).abs() < 0.02, "got {ratio}");
    }

    #[test]
    fn test_rng_stays_in_unit_interval() {
        let mut rng = Rng::new(0);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "{v} out of range");
        }
    }
}
