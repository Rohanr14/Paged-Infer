//! Turning a growing token stream into a growing text stream.
//!
//! Detokenizing one token at a time does not work, for two reasons that both
//! produce visible corruption rather than a clean error:
//!
//! * A BPE token can carry a *fragment* of a UTF-8 character. Emoji and CJK
//!   text routinely split across two to four tokens, and decoding the first
//!   ones alone yields replacement characters — one per byte, with the
//!   byte-fallback decoder Llama tokenizers use.
//! * SentencePiece encodes whitespace into the token itself (`▁the`), and
//!   whether that leading space survives depends on what came before. Decoding
//!   `["▁the"]` in isolation and decoding it as part of a sentence do not agree.
//!
//! So the whole prefix is decoded each time and only the newly-appeared text is
//! emitted. That is quadratic in the number of tokens, which sounds alarming and
//! is not: decoding a few hundred ids is microseconds against a forward pass
//! measured in milliseconds, and the arithmetic is dwarfed by the model.
//!
//! Re-decoding alone is not enough, though. Text already sent over a socket
//! cannot be taken back, so what is sent has to be *stable*: text the decoder
//! could still revise once more tokens arrive is held until it cannot. The
//! only revisable suffix a byte-fallback or byte-level decoder produces is a
//! run of U+FFFD replacement characters standing in for an incomplete UTF-8
//! sequence — three of them for the first three bytes of an emoji, replaced by
//! the emoji itself when the fourth byte lands. So a trailing run of U+FFFD is
//! never emitted while more tokens may come; a genuine U+FFFD in the text is
//! simply delayed until the next token, or until [`IncrementalDetokenizer::finish`]
//! flushes it. Either way the concatenation of every delta and the flush is
//! exactly the decode of the whole token list.
//!
//! One more property of the byte-fallback decoder matters: it converts a whole
//! run of consecutive byte tokens at once, so `你` followed by the first byte
//! of `好` decodes to four replacement characters, not to `你` plus one. The
//! stable prefix can therefore shrink for a step and grow back. What was sent
//! is tracked as exactly that — the concatenation of every delta, which only
//! ever grows — and each new delta is whatever the stable text has beyond it.

use tokenizers::Tokenizer;

/// Text that appeared in `full` but was not yet in `emitted`.
///
/// Normally `full` simply extends `emitted` and this is the new suffix. It is
/// not guaranteed to: a tokenizer may revise the tail it produced earlier once
/// the following token disambiguates it. When that happens the two strings are
/// cut at their longest common prefix, so the stream re-syncs on the corrected
/// text instead of stalling or repeating what was already sent.
pub fn new_text<'a>(emitted: &str, full: &'a str) -> &'a str {
    if let Some(rest) = full.strip_prefix(emitted) {
        return rest;
    }
    let common: usize = emitted
        .chars()
        .zip(full.chars())
        .take_while(|(a, b)| a == b)
        .map(|(a, _)| a.len_utf8())
        .sum();
    &full[common..]
}

/// The part of `decoded` that no further token can change: everything up to a
/// trailing run of U+FFFD, which may be an incomplete multi-byte character.
pub fn stable_prefix(decoded: &str) -> &str {
    decoded.trim_end_matches('\u{FFFD}')
}

/// Accumulates token ids and hands back the text each batch of them revealed.
#[derive(Debug, Default)]
pub struct IncrementalDetokenizer {
    tokens: Vec<u32>,
    /// The decode of every token so far, including any unstable tail.
    decoded: String,
    /// What has been handed out: the concatenation of every delta. Never
    /// shrinks, even when the decoder's own text momentarily does.
    emitted: String,
}

impl IncrementalDetokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append tokens and return the stable text they added.
    ///
    /// Returns an empty string when the new tokens do not complete a character
    /// yet — the caller should send nothing rather than send a placeholder, and
    /// the text will arrive with the token that finishes it.
    pub fn push(&mut self, tokenizer: &Tokenizer, tokens: &[u32]) -> String {
        self.tokens.extend_from_slice(tokens);
        let Ok(full) = tokenizer.decode(&self.tokens, true) else {
            return String::new();
        };
        self.decoded = full;
        let delta = new_text(&self.emitted, stable_prefix(&self.decoded)).to_string();
        self.emitted.push_str(&delta);
        delta
    }

    /// Release whatever was being held back. Call once, when no more tokens
    /// will come: a replacement character at the very end of the output is
    /// then the decoder's final word rather than a fragment.
    pub fn finish(&mut self) -> String {
        let delta = new_text(&self.emitted, &self.decoded).to_string();
        self.emitted.push_str(&delta);
        delta
    }

    /// Everything emitted so far.
    pub fn text(&self) -> &str {
        &self.emitted
    }

    /// The full decode, held-back tail included.
    pub fn decoded(&self) -> &str {
        &self.decoded
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_extension_yields_the_suffix() {
        assert_eq!(new_text("hello", "hello world"), " world");
        assert_eq!(new_text("", "hi"), "hi");
        assert_eq!(new_text("same", "same"), "");
    }

    #[test]
    fn a_revised_tail_resyncs_at_the_common_prefix() {
        // The tokenizer decided the trailing text was something else once the
        // next token arrived. Emitting from the divergence point is correct;
        // emitting the whole string again would duplicate what the client has.
        assert_eq!(new_text("the cat sao", "the cat sat"), "t");
        assert_eq!(new_text("abc", "abd"), "d");
        // A shorter revision is still handled: nothing is emitted, and the next
        // push diffs against the corrected text.
        assert_eq!(new_text("abcd", "abc"), "");
    }

    #[test]
    fn resync_never_splits_a_multibyte_character() {
        // The common prefix is measured in characters and converted to bytes,
        // so slicing can never land inside a UTF-8 sequence.
        let prev = "héllo wörld";
        let full = "héllo wÖrld!";
        let delta = new_text(prev, full);
        assert!(full.ends_with(delta));
        assert_eq!(delta, "Örld!");
    }

    #[test]
    fn concatenating_every_delta_reproduces_the_final_text() {
        // The property a streaming client depends on.
        let steps = ["", "Th", "The ", "The qu", "The quick", "The quick fox"];
        let mut emitted = String::new();
        let mut joined = String::new();
        for full in steps {
            joined.push_str(new_text(&emitted, full));
            emitted = full.to_string();
        }
        assert_eq!(joined, "The quick fox");
    }

    #[test]
    fn the_stable_prefix_holds_back_only_a_trailing_replacement_run() {
        assert_eq!(stable_prefix("abc"), "abc");
        assert_eq!(stable_prefix("abc\u{FFFD}"), "abc");
        assert_eq!(stable_prefix("abc\u{FFFD}\u{FFFD}\u{FFFD}"), "abc");
        // A replacement character followed by real text is final.
        assert_eq!(stable_prefix("a\u{FFFD}b"), "a\u{FFFD}b");
        assert_eq!(stable_prefix("\u{FFFD}"), "");
        assert_eq!(stable_prefix(""), "");
    }
}
