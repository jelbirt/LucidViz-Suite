//! Stop-word filtering using the `stop-words` crate.

use std::collections::HashSet;

use crate::types::Token;

/// Remove stop-words from `tokens` using the specified BCP 47 language tag.
///
/// Issues a warning if the language is not supported and returns the unfiltered
/// token list. Uses `catch_unwind` because the `stop-words` crate panics on
/// unknown language codes rather than returning an empty list.
pub fn filter_stopwords(tokens: Vec<Token>, language: &str) -> Vec<Token> {
    // stop_words::get() panics on unsupported language codes, so we catch it.
    let owned_lang = language.to_string();
    let result = std::panic::catch_unwind(move || stop_words::get(&owned_lang));
    let sw_list = match result {
        Ok(list) => list,
        Err(_) => {
            eprintln!(
                "[mf-pipeline] Warning: no stop-words found for language '{language}'; skipping filter"
            );
            return tokens;
        }
    };

    if sw_list.is_empty() {
        eprintln!(
            "[mf-pipeline] Warning: no stop-words found for language '{language}'; skipping filter"
        );
        return tokens;
    }

    let sw_set: HashSet<String> = sw_list.iter().map(|s| s.to_string()).collect();
    tokens
        .into_iter()
        .filter(|t| !sw_set.contains(t.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stopwords_english() {
        let tokens: Vec<Token> = ["the", "quick", "brown", "fox", "is", "a", "animal"]
            .iter()
            .map(|s| Token(s.to_string()))
            .collect();
        let filtered = filter_stopwords(tokens, "en");
        let words: Vec<&str> = filtered.iter().map(|t| t.as_str()).collect();
        // "the", "is", "a" are common English stop-words.
        assert!(!words.contains(&"the"), "'the' should be filtered");
        assert!(!words.contains(&"is"), "'is' should be filtered");
        assert!(words.contains(&"fox"), "'fox' should remain");
    }

    #[test]
    fn test_stopwords_unknown_language_passthrough() {
        let tokens: Vec<Token> = vec![Token("hello".to_string()), Token("world".to_string())];
        let filtered = filter_stopwords(tokens.clone(), "xx-unknown");
        // Should return original tokens unchanged.
        assert_eq!(filtered.len(), tokens.len());
    }
}
