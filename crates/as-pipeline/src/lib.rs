//! `as-pipeline` — AlignSpace: SE matrix, MDS, Procrustes, centrality.
//!
//! See `implementation_plan.md` §5 for full specification.

pub mod centrality;
pub mod error;
pub mod mds;
pub mod normalize;
pub mod output;
pub mod pipeline;
pub mod procrustes;
pub mod structural_eq;
pub mod types;

pub use error::AsError;
pub use types::*;
