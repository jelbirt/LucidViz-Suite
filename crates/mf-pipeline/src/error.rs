//! Error types for the MatrixForge pipeline.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("input '{path}' is {bytes} bytes, exceeding limit of {limit} bytes")]
    FileTooLarge {
        path: String,
        bytes: u64,
        limit: u64,
    },

    #[error("Empty corpus: no text was ingested")]
    EmptyCorpus,

    #[error("Vocabulary is empty after filtering (min_count too high?)")]
    EmptyVocabulary,

    #[error("XLSX write error: {0}")]
    Xlsx(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Unknown language for stop-words: {0}")]
    UnknownLanguage(String),
}
