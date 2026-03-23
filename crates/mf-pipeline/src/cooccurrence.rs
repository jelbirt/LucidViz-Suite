//! Sliding-window co-occurrence counting.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::types::{CooccurrenceGraph, MfConfig, Token};

/// Build a co-occurrence matrix from a token stream.
///
/// Only tokens whose unigram count is ≥ `config.min_count` are included in the
/// vocabulary.  The vocabulary is sorted alphabetically for determinism.
///
/// The sliding window advances by `config.slide_rate` tokens per step.
pub fn build_cooccurrence(tokens: &[Token], config: &MfConfig) -> CooccurrenceGraph {
    // Count unigrams to determine vocabulary.
    let mut unigram_counts: HashMap<&str, u64> = HashMap::new();
    for t in tokens {
        *unigram_counts.entry(t.as_str()).or_insert(0) += 1;
    }

    // Vocabulary: tokens with count >= min_count, sorted.
    let mut vocab: Vec<Token> = unigram_counts
        .iter()
        .filter(|(_, &count)| count >= config.min_count)
        .map(|(word, _)| Token(word.to_string()))
        .collect();
    vocab.sort_by(|a, b| a.0.cmp(&b.0));

    if vocab.is_empty() {
        return CooccurrenceGraph::new(vocab, config.window_size, config.slide_rate);
    }

    let n = vocab.len();
    build_cooccurrence_for_vocab(tokens, vocab, n, config.window_size, config.slide_rate)
}

/// Build a co-occurrence matrix using an already fixed/shared vocabulary.
pub fn build_cooccurrence_with_vocab(
    tokens: &[Token],
    vocab: &[Token],
    config: &MfConfig,
) -> CooccurrenceGraph {
    build_cooccurrence_for_vocab(
        tokens,
        vocab.to_vec(),
        vocab.len(),
        config.window_size,
        config.slide_rate,
    )
}

fn build_cooccurrence_for_vocab(
    tokens: &[Token],
    vocab: Vec<Token>,
    n: usize,
    window: usize,
    slide_rate: usize,
) -> CooccurrenceGraph {
    if vocab.is_empty() {
        return CooccurrenceGraph::new(vocab, window, slide_rate);
    }

    let slide = slide_rate.max(1);
    let num_tokens = tokens.len();

    // Build vocab index for O(1) lookup.
    let vocab_index: HashMap<&str, usize> = vocab
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();

    // Determine chunk boundaries for parallelism.
    // Each chunk covers a window of `2*window + slide` tokens so adjacent
    // chunks have enough context.  We then sum the partial matrices.
    let chunk_size = (num_tokens / rayon::current_num_threads()).max(window * 2 + slide);

    // Build per-chunk partial matrices, then reduce.
    let chunks: Vec<&[Token]> = {
        let mut v = Vec::new();
        let mut start = 0;
        while start < num_tokens {
            let end = (start + chunk_size + window).min(num_tokens);
            v.push(&tokens[start..end]);
            if start + chunk_size >= num_tokens {
                break;
            }
            start += chunk_size;
        }
        v
    };

    // Starting positions for each chunk (to correctly slide the window).
    let chunk_starts: Vec<usize> = {
        let mut v = vec![0usize];
        let mut start = 0usize;
        for _ in 0..chunks.len().saturating_sub(1) {
            start += chunk_size;
            v.push(start);
        }
        v
    };

    let partial_matrices: Vec<HashMap<usize, u64>> = chunks
        .par_iter()
        .zip(chunk_starts.par_iter())
        .map(|(chunk, &chunk_start)| {
            let mut mat = HashMap::<usize, u64>::new();
            let len = chunk.len();
            let local_center_end = (chunk_size).min(len);
            let mut pos = if slide > 1 {
                (slide - (chunk_start % slide)) % slide
            } else {
                0
            };
            while pos < local_center_end {
                if let Some(&ci) = vocab_index.get(chunk[pos].as_str()) {
                    // Look left and right within the window.
                    let left = pos.saturating_sub(window);
                    let right = (pos + window + 1).min(len);
                    for other_pos in left..right {
                        if other_pos == pos {
                            continue;
                        }
                        if let Some(&cj) = vocab_index.get(chunk[other_pos].as_str()) {
                            *mat.entry(ci * n + cj).or_insert(0) += 1;
                        }
                    }
                }
                pos += slide;
            }
            mat
        })
        .collect();

    // Sum partial matrices.
    let mut total_mat = vec![0u64; n * n];
    for partial in partial_matrices {
        for (i, v) in partial {
            total_mat[i] += v;
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let mirrored = total_mat[i * n + j].max(total_mat[j * n + i]);
            total_mat[i * n + j] = mirrored;
            total_mat[j * n + i] = mirrored;
        }
    }

    let mut graph = CooccurrenceGraph::new(vocab, window, slide);
    graph.matrix = total_mat;
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> MfConfig {
        MfConfig {
            window_size: 2,
            slide_rate: 1,
            min_count: 1,
            ..MfConfig::default()
        }
    }

    fn tokens(words: &[&str]) -> Vec<Token> {
        words.iter().map(|w| Token(w.to_string())).collect()
    }

    #[test]
    fn test_cooccurrence_known_sequence() {
        // a b c a b — "a" and "b" co-occur most.
        let toks = tokens(&["a", "b", "c", "a", "b"]);
        let cfg = make_config();
        let graph = build_cooccurrence(&toks, &cfg);

        // Vocabulary sorted: a, b, c
        assert_eq!(graph.vocab_size, 3);
        let vocab_words: Vec<&str> = graph.vocab.iter().map(|t| t.as_str()).collect();
        let a = vocab_words.iter().position(|&w| w == "a").unwrap();
        let b = vocab_words.iter().position(|&w| w == "b").unwrap();

        // a and b co-occur — matrix should be symmetric.
        assert!(graph.get(a, b) > 0, "a,b should co-occur");
        assert_eq!(
            graph.get(a, b),
            graph.get(b, a),
            "matrix should be symmetric"
        );
    }

    #[test]
    fn test_cooccurrence_min_count_filters() {
        // "rare" appears only once; with min_count=2 it should be excluded.
        let toks = tokens(&["hello", "world", "hello", "world", "rare"]);
        let cfg = MfConfig {
            min_count: 2,
            ..MfConfig::default()
        };
        let graph = build_cooccurrence(&toks, &cfg);
        assert!(
            graph.vocab.iter().all(|t| t.as_str() != "rare"),
            "'rare' should be filtered by min_count=2"
        );
    }

    #[test]
    fn test_cooccurrence_chunk_boundaries_match_reference() {
        let toks = tokens(&["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]);
        let cfg = MfConfig {
            window_size: 2,
            slide_rate: 2,
            min_count: 1,
            ..MfConfig::default()
        };
        let graph = build_cooccurrence(&toks, &cfg);

        let vocab_words: Vec<&str> = graph.vocab.iter().map(|t| t.as_str()).collect();
        let g = vocab_words.iter().position(|&w| w == "g").unwrap();
        let h = vocab_words.iter().position(|&w| w == "h").unwrap();

        assert_eq!(graph.get(g, h), 1, "boundary pairs should not double-count");
        assert_eq!(graph.get(h, g), 1, "matrix should remain symmetric");
    }
}
