//! `mf-pipeline` — MatrixForge: text analysis → co-occurrence network → PMI.
//!
//! See `implementation_plan.md` §3 / Phase 3 for full specification.

pub mod centrality;
pub mod cooccurrence;
pub mod error;
pub mod graph;
#[cfg(feature = "native-io")]
pub mod ingest;
pub mod normalize;
#[cfg(feature = "native-io")]
pub mod output;
#[cfg(feature = "native-io")]
pub mod pipeline;
pub mod pmi;
pub mod stopwords;
pub mod svd_similarity;
pub mod tokenize;
pub mod types;

pub use error::MfError;
pub use types::*;
