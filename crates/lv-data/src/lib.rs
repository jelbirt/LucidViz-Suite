#![allow(dead_code)]

//! `lv-data` — shared data model and XLSX/JSON I/O for the Lucid Visualization Suite.

pub mod error;
pub mod schema;
pub mod validation;

#[cfg(feature = "native-io")]
pub mod json_io;
#[cfg(feature = "native-io")]
pub mod xlsx_reader;
#[cfg(feature = "native-io")]
pub mod xlsx_writer;

pub use error::DataError;
#[cfg(feature = "native-io")]
pub use json_io::{read_etv_json as load_dataset_json, write_etv_json};
pub use schema::{
    EdgeRow, EtvDataset, EtvRow, EtvSheet, GpuInstance, LisBuffer, LisConfig, LisFrame, ShapeKind,
};
pub use validation::validate_dataset;
#[cfg(feature = "native-io")]
pub use xlsx_reader::read_etv_xlsx;
