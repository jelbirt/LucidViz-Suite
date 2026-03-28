//! Stop-word filtering using the `stop-words` crate.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crate::types::Token;

/// Cached stop-word sets keyed by language code.
static STOPWORD_CACHE: std::sync::LazyLock<Mutex<HashMap<String, Option<HashSet<String>>>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Remove stop-words from `tokens` using the specified BCP 47 language tag.
///
/// Issues a warning if the language is not supported and returns the unfiltered
/// token list. Uses `catch_unwind` because the `stop-words` crate panics on
/// unknown language codes rather than returning an empty list.
///
/// The stop-word set is cached across calls to avoid rebuilding on every invocation.
pub fn filter_stopwords(tokens: Vec<Token>, language: &str) -> Vec<Token> {
    let sw_set = {
        let mut cache = STOPWORD_CACHE.lock().unwrap();
        cache
            .entry(language.to_string())
            .or_insert_with(|| {
                let owned_lang = language.to_string();
                let result =
                    std::panic::catch_unwind(move || stop_words::get(&owned_lang));
                match result {
                    Ok(list) if !list.is_empty() => {
                        Some(list.iter().map(|s| s.to_string()).collect())
                    }
                    _ => None,
                }
            })
            .clone()
    };

    let Some(sw_set) = sw_set else {
        eprintln!(
            "[mf-pipeline] Warning: no stop-words found for language '{language}'; skipping filter"
        );
        return tokens;
    };

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
