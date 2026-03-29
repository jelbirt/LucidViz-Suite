//! `lv-data` — shared data model and XLSX/JSON I/O for the Lucid Visualization Suite.

pub mod analysis;
pub mod bridge;
pub mod error;
pub mod io_util;
pub mod schema;
pub mod validation;

#[cfg(feature = "native-io")]
pub mod json_io;
#[cfg(feature = "native-io")]
pub mod xlsx_reader;
#[cfg(feature = "native-io")]
pub mod xlsx_writer;

pub use analysis::CentralityReport;
pub use bridge::SimToDistMethod;
pub use error::DataError;
#[cfg(feature = "native-io")]
pub use json_io::{read_lv_json as load_dataset_json, write_lv_json};
pub use schema::{
    EdgeRow, GpuInstance, LisBuffer, LisConfig, LisFrame, LvDataset, LvRow, LvSheet, ShapeKind,
};
pub use validation::validate_dataset;
#[cfg(feature = "native-io")]
pub use xlsx_reader::read_lv_xlsx;
