//! Speculative decoding: guess several tokens cheaply, then check them all in
//! one model pass.
//!
//! Decoding is memory-bound — a step streams every weight matrix to produce one
//! token — so a pass that produces one token and a pass that checks `K+1`
//! positions cost nearly the same. If a cheap guesser proposes `K` tokens and
//! the model confirms `j` of them, the sequence advances `j+1` tokens for the
//! price of roughly one step. That is the one optimization here that helps a
//! *single* sequence: batching across sequences does nothing for one user
//! waiting on one stream.
//!
//! **This is lossless under greedy decoding, not an approximation.** A draft
//! token is accepted only when it equals what the model itself would have
//! produced at that position, and the token after the last accepted one is
//! taken from the model's own logits. So the emitted sequence is exactly the
//! greedy sequence — verified in `tests/speculative_tests.rs`, which compares
//! speculative output against greedy output token for token.
//!
//! The drafters here need no second model. That matters: a draft model would
//! mean another checkpoint to ship, load and keep resident, and at 1B scale the
//! draft model's own memory traffic eats the win.

use std::collections::HashMap;

/// Something that proposes continuation tokens without running the model.
pub trait Drafter: Send {
    /// Guess up to `k` tokens that follow `context`. Returning fewer (or none)
    /// is always allowed — the engine falls back to an ordinary decode step.
    fn draft(&mut self, context: &[u32], k: usize) -> Vec<u32>;

    /// Called with the full sequence after each step, so stateful drafters can
    /// learn. Stateless ones ignore it.
    fn observe(&mut self, _context: &[u32]) {}

    fn name(&self) -> &'static str;
}

/// Propose the continuation that followed the last time this suffix appeared.
///
/// Serving workloads copy constantly: summarize this document, answer from this
/// passage, rewrite this function, fix this stack trace. In all of them long
/// spans of the output already appear verbatim in the input, so the cheapest
/// possible draft — "what came after this the last time we saw it" — is often
/// right for many tokens in a row.
///
/// Matching prefers the longest suffix that occurs earlier, since a longer
/// match is stronger evidence, and scans from the end backwards so the most
/// recent occurrence wins.
#[derive(Debug, Clone)]
pub struct PromptLookupDrafter {
    min_ngram: usize,
    max_ngram: usize,
}

impl Default for PromptLookupDrafter {
    fn default() -> Self {
        Self::new(2, 8)
    }
}

impl PromptLookupDrafter {
    pub fn new(min_ngram: usize, max_ngram: usize) -> Self {
        assert!(min_ngram >= 1, "an n-gram needs at least one token");
        Self {
            min_ngram,
            max_ngram: max_ngram.max(min_ngram),
        }
    }
}

impl Drafter for PromptLookupDrafter {
    fn draft(&mut self, context: &[u32], k: usize) -> Vec<u32> {
        if k == 0 || context.len() <= self.min_ngram {
            return Vec::new();
        }
        let max_n = self.max_ngram.min(context.len() - 1);

        for n in (self.min_ngram..=max_n).rev() {
            let suffix = &context[context.len() - n..];
            // Search backwards: the most recent occurrence is the most likely
            // to continue the same way.
            let limit = context.len() - n;
            for start in (0..limit).rev() {
                if &context[start..start + n] == suffix {
                    let from = start + n;
                    if from >= limit {
                        // The match runs into the suffix itself; nothing to copy.
                        continue;
                    }
                    let take = k.min(limit - from);
                    if take > 0 {
                        return context[from..from + take].to_vec();
                    }
                }
            }
        }
        Vec::new()
    }

    fn name(&self) -> &'static str {
        "prompt-lookup"
    }
}

/// N-gram based drafter: remembers which token most often followed each n-gram
/// it has seen, and proposes by walking that table forward.
///
/// Weaker than [`PromptLookupDrafter`] on copy-heavy work, but it keeps
/// learning across a long generation rather than only mining the context.
pub struct NgramDrafter {
    n: usize,
    table: HashMap<Vec<u32>, (u32, u32)>, // key -> (best_token, count)
    history: Vec<u32>,
}

impl NgramDrafter {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            table: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Record that `next_token` followed the current history context.
    pub fn observe_token(&mut self, next_token: u32) {
        if self.history.len() >= self.n {
            let key: Vec<u32> = self.history[self.history.len() - self.n..].to_vec();
            let entry = self.table.entry(key).or_insert((next_token, 0));
            if next_token == entry.0 {
                entry.1 += 1;
            } else if entry.1 == 0 {
                *entry = (next_token, 1);
            } else {
                entry.1 -= 1;
            }
        }
        self.history.push(next_token);
    }
}

impl Drafter for NgramDrafter {
    fn draft(&mut self, context: &[u32], k: usize) -> Vec<u32> {
        // Catch up on anything the engine appended since the last call.
        if context.len() > self.history.len() {
            let start = self.history.len();
            for &t in &context[start..] {
                self.observe_token(t);
            }
        }

        let mut drafts = Vec::with_capacity(k);
        if context.len() < self.n {
            return drafts;
        }
        let mut ctx: Vec<u32> = context[context.len() - self.n..].to_vec();
        for _ in 0..k {
            match self.table.get(&ctx) {
                Some(&(token, _)) => {
                    drafts.push(token);
                    ctx.remove(0);
                    ctx.push(token);
                }
                None => break,
            }
        }
        drafts
    }

    fn name(&self) -> &'static str {
        "ngram"
    }
}

/// Which draft tokens the model confirmed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    /// Draft tokens the model agreed with, in order.
    pub accepted: Vec<u32>,
    /// The token the model produced at the first position it was consulted
    /// about beyond the accepted run. Always emitted, so a step never stalls
    /// even when every draft is wrong.
    pub corrected: u32,
    pub drafted: usize,
}

impl Verdict {
    /// Tokens this step appends to the sequence.
    pub fn emitted(&self) -> usize {
        self.accepted.len() + 1
    }
}

/// Compare drafts against what the model actually predicted.
///
/// `model_argmax[i]` is the model's own choice at the position that draft
/// `i` was proposed for; there is always one more of these than there are
/// drafts, because the verification pass also scores the position after the
/// last draft.
///
/// Accepting stops at the first disagreement: once a draft is wrong, every
/// position after it was conditioned on a token the model would not have
/// produced, so its prediction is meaningless.
pub fn verify_greedy(drafts: &[u32], model_argmax: &[u32]) -> Verdict {
    assert_eq!(
        model_argmax.len(),
        drafts.len() + 1,
        "verification needs one model prediction per draft, plus one more"
    );

    let accepted: Vec<u32> = drafts
        .iter()
        .zip(model_argmax.iter())
        .take_while(|(d, m)| d == m)
        .map(|(d, _)| *d)
        .collect();

    Verdict {
        corrected: model_argmax[accepted.len()],
        accepted,
        drafted: drafts.len(),
    }
}

/// Running tally of how well drafting is doing.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStats {
    pub steps: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// Tokens emitted, including the corrected token from each step.
    pub emitted: u64,
    /// Verification passes that ran, i.e. model forward passes.
    pub passes: u64,
}

impl SpecStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }

    /// Tokens emitted per sequence-step. Ordinary decoding is exactly 1.0, so
    /// this is what speculation bought, independent of how many sequences were
    /// in flight.
    ///
    /// The wall-clock gain is lower, because a pass verifying `K+1` positions
    /// costs somewhat more than one producing a single token.
    pub fn tokens_per_step(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.emitted as f64 / self.steps as f64
        }
    }

    /// Tokens emitted per batched model pass.
    ///
    /// Beware when comparing runs: this counts *every* sequence in the batch,
    /// so with `B` sequences in flight the no-speculation baseline is `B`,
    /// not 1. Use [`SpecStats::tokens_per_step`] to isolate speculation from
    /// cross-sequence batching.
    pub fn tokens_per_pass(&self) -> f64 {
        if self.passes == 0 {
            0.0
        } else {
            self.emitted as f64 / self.passes as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_accepts_a_matching_run_and_stops_at_the_first_miss() {
        let v = verify_greedy(&[10, 11, 12], &[10, 11, 99, 13]);
        assert_eq!(v.accepted, vec![10, 11]);
        assert_eq!(v.corrected, 99);
        assert_eq!(v.emitted(), 3);
    }

    #[test]
    fn test_verify_accepts_everything_when_the_draft_is_perfect() {
        let v = verify_greedy(&[1, 2, 3], &[1, 2, 3, 4]);
        assert_eq!(v.accepted, vec![1, 2, 3]);
        assert_eq!(v.corrected, 4);
        assert_eq!(v.emitted(), 4);
    }

    #[test]
    fn test_verify_still_emits_a_token_when_every_draft_is_wrong() {
        // The floor: a completely useless drafter costs accuracy nothing and
        // still advances one token per pass, exactly like ordinary decoding.
        let v = verify_greedy(&[7, 7, 7], &[1, 2, 3, 4]);
        assert!(v.accepted.is_empty());
        assert_eq!(v.corrected, 1);
        assert_eq!(v.emitted(), 1);
    }

    #[test]
    fn test_verify_with_no_drafts_is_an_ordinary_step() {
        let v = verify_greedy(&[], &[42]);
        assert!(v.accepted.is_empty());
        assert_eq!(v.corrected, 42);
        assert_eq!(v.emitted(), 1);
    }

    #[test]
    fn test_prompt_lookup_continues_a_repeated_span() {
        let mut d = PromptLookupDrafter::new(2, 8);
        // "the quick brown fox ... the quick" -> should propose "brown fox".
        let ctx = vec![1, 2, 3, 4, 5, 9, 9, 1, 2];
        assert_eq!(d.draft(&ctx, 3), vec![3, 4, 5]);
    }

    #[test]
    fn test_prompt_lookup_prefers_the_longest_match() {
        let mut d = PromptLookupDrafter::new(1, 8);
        // Suffix [2,3] occurs earlier followed by 4; bare [3] also occurs
        // followed by 4. The longer match should win and both agree here.
        let ctx = vec![2, 3, 4, 7, 8, 2, 3];
        assert_eq!(d.draft(&ctx, 2), vec![4, 7]);
    }

    #[test]
    fn test_prompt_lookup_returns_nothing_when_nothing_repeats() {
        let mut d = PromptLookupDrafter::new(2, 8);
        assert!(d.draft(&[1, 2, 3, 4, 5], 4).is_empty());
    }

    #[test]
    fn test_prompt_lookup_handles_tiny_contexts() {
        let mut d = PromptLookupDrafter::new(2, 8);
        assert!(d.draft(&[], 4).is_empty());
        assert!(d.draft(&[1], 4).is_empty());
        assert!(d.draft(&[1, 2], 0).is_empty());
    }

    #[test]
    fn test_prompt_lookup_never_proposes_more_than_asked() {
        let mut d = PromptLookupDrafter::new(2, 8);
        let ctx: Vec<u32> = (0..40).chain(0..4).collect();
        for k in 0..8 {
            assert!(d.draft(&ctx, k).len() <= k);
        }
    }

    #[test]
    fn test_ngram_learns_a_repeating_cycle() {
        let mut d = NgramDrafter::new(2);
        let ctx: Vec<u32> = [1, 2, 3].repeat(6);
        let drafts = d.draft(&ctx, 3);
        assert_eq!(drafts, vec![1, 2, 3], "should have learned the cycle");
    }

    #[test]
    fn test_stats_arithmetic() {
        let s = SpecStats {
            steps: 10,
            drafted: 40,
            accepted: 24,
            emitted: 34,
            passes: 10,
        };
        assert!((s.acceptance_rate() - 0.6).abs() < 1e-9);
        assert!((s.tokens_per_step() - 3.4).abs() < 1e-9);
        assert!((s.tokens_per_pass() - 3.4).abs() < 1e-9);
    }
}
