use std::collections::HashMap;

/// N-gram based speculative drafter.
/// Maintains a table of (last N tokens) -> most-frequently-observed next token.
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
    pub fn observe(&mut self, next_token: u32) {
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

    /// Draft up to `k` speculative tokens given the current history.
    /// Returns a Vec of predicted tokens (may be shorter than k if history is too short).
    pub fn draft(&self, k: usize) -> Vec<u32> {
        let mut drafts = Vec::with_capacity(k);
        if self.history.len() < self.n {
            return drafts;
        }
        let mut ctx: Vec<u32> = self.history[self.history.len() - self.n..].to_vec();
        for _ in 0..k {
            if let Some(&(token, _)) = self.table.get(&ctx) {
                drafts.push(token);
                ctx.remove(0);
                ctx.push(token);
            } else {
                break;
            }
        }
        drafts
    }
}

/// Result of one speculative decoding step.
pub struct SpecResult {
    pub accepted_tokens: Vec<u32>,
    pub corrected_token: u32,
    pub draft_count: usize,
    pub accept_count: usize,
}
