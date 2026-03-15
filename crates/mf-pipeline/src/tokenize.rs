//! Tokenization: split normalized text into `Token`s.
//!
//! Filters:
//!  - Purely numeric tokens are removed.
//!  - Single-character tokens are removed.

use unicode_segmentation::UnicodeSegmentation;

use crate::types::Token;

/// Split `text` into tokens, filtering numeric-only and single-char tokens.
///
/// `text` should already be normalized (lowercase, NFC, no stray punctuation).
pub fn tokenize(text: &str) -> Vec<Token> {
    text.unicode_words()
        .filter(|w| {
            // Filter single-char tokens.
            let char_count = w.chars().count();
            if char_count < 2 {
                return false;
            }
            // Filter purely numeric tokens.
            if w.chars().all(|c| c.is_ascii_digit()) {
                return false;
            }
            true
        })
        .map(|w| Token(w.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_ascii_sentence() {
        let tokens = tokenize("the quick brown fox jumps");
        let words: Vec<&str> = tokens.iter().map(|t| t.as_str()).collect();
        assert!(words.contains(&"the"));
        assert!(words.contains(&"quick"));
        assert!(words.contains(&"fox"));
    }

    #[test]
    fn test_tokenize_filters_numeric() {
        let tokens = tokenize("we have 42 apples and 100 oranges");
        assert!(
            tokens
                .iter()
                .all(|t| !t.0.chars().all(|c| c.is_ascii_digit())),
            "numeric tokens should be filtered"
        );
    }

    #[test]
    fn test_tokenize_filters_single_char() {
        let tokens = tokenize("a b c hello world");
        let words: Vec<&str> = tokens.iter().map(|t| t.as_str()).collect();
        assert!(
            !words.contains(&"a"),
            "single-char tokens should be filtered"
        );
        assert!(words.contains(&"hello"));
    }

    #[test]
    fn test_tokenize_cjk_mixed() {
        // CJK characters — unicode_words should handle them.
        let tokens = tokenize("hello 世界 world");
        // At minimum "hello" and "world" should be present.
        let words: Vec<&str> = tokens.iter().map(|t| t.as_str()).collect();
        assert!(words.contains(&"hello"));
        assert!(words.contains(&"world"));
    }
}
