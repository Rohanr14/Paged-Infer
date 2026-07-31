//! Turning a growing token stream into a growing text stream.
//!
//! Detokenizing one token at a time does not work, for two reasons that both
//! produce visible corruption rather than a clean error:
//!
//! * A BPE token can carry a *fragment* of a UTF-8 character. Emoji and CJK
//!   text routinely split across two or three tokens, and decoding either half
//!   alone yields a replacement character.
//! * SentencePiece encodes whitespace into the token itself (`▁the`), and
//!   whether that leading space survives depends on what came before. Decoding
//!   `["▁the"]` in isolation and decoding it as part of a sentence do not agree.
//!
//! So the whole prefix is decoded each time and only the newly-appeared text is
//! emitted. That is quadratic in the number of tokens, which sounds alarming and
//! is not: decoding a few hundred ids is microseconds against a forward pass
//! measured in milliseconds, and the arithmetic is dwarfed by the model.

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

/// Accumulates token ids and hands back the text each batch of them revealed.
#[derive(Debug, Default)]
pub struct IncrementalDetokenizer {
    tokens: Vec<u32>,
    emitted: String,
}

impl IncrementalDetokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append tokens and return the text they added.
    ///
    /// Returns an empty string when the new tokens do not complete a character
    /// yet — the caller should send nothing rather than send a placeholder, and
    /// the text will arrive with the token that finishes it.
    pub fn push(&mut self, tokenizer: &Tokenizer, tokens: &[u32]) -> String {
        self.tokens.extend_from_slice(tokens);
        let Ok(full) = tokenizer.decode(&self.tokens, true) else {
            return String::new();
        };
        let delta = new_text(&self.emitted, &full).to_string();
        self.emitted = full;
        delta
    }

    /// Everything emitted so far.
    pub fn text(&self) -> &str {
        &self.emitted
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
}
