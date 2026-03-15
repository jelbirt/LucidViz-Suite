//! Unicode text normalization for MF pipeline input.
//!
//! Steps (in order):
//!  1. NFC Unicode normalization
//!  2. Case-fold to lowercase
//!  3. Strip characters that are neither alphanumeric nor whitespace
//!  4. Collapse runs of whitespace to a single space
//!  5. Trim leading/trailing whitespace

use unicode_normalization::UnicodeNormalization;

/// Normalize raw text for tokenization.
pub fn normalize_text(raw: &str) -> String {
    // Step 1: NFC normalization.
    let nfc: String = raw.nfc().collect();

    // Steps 2-3: Lowercase and strip non-alphanumeric non-whitespace.
    let filtered: String = nfc
        .chars()
        .flat_map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                // Lowercase each char (may expand to multiple chars for some Unicode).
                c.to_lowercase().collect::<Vec<_>>()
            } else {
                // Replace with a space so adjacent words don't merge.
                vec![' ']
            }
        })
        .collect();

    // Steps 4-5: Collapse whitespace and trim.
    filtered.split_whitespace().collect::<Vec<&str>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_case_fold() {
        assert_eq!(normalize_text("Hello World"), "hello world");
    }

    #[test]
    fn test_normalize_nfc_applied() {
        // "é" as combining e + combining acute (NFD) → single char (NFC).
        let nfd = "e\u{0301}"; // e + combining acute
        let result = normalize_text(nfd);
        assert_eq!(result, "\u{00e9}"); // é as single NFC codepoint
    }

    #[test]
    fn test_normalize_punctuation_stripped() {
        let result = normalize_text("hello, world! how's it going?");
        // Apostrophe and punctuation stripped; extra space collapsed.
        assert!(!result.contains(','), "comma should be stripped");
        assert!(!result.contains('!'), "exclamation should be stripped");
        assert!(!result.contains('?'), "question mark should be stripped");
    }

    #[test]
    fn test_normalize_whitespace_collapse() {
        let result = normalize_text("  hello   world  ");
        assert_eq!(result, "hello world");
    }
}
