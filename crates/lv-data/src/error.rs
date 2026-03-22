use thiserror::Error;

/// All errors produced by the `lv-data` crate.
#[derive(Debug, Error)]
pub enum DataError {
    // ── I/O ─────────────────────────────────────────────────────────────────
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("XLSX read error: {0}")]
    XlsxRead(#[from] calamine::Error),

    #[error("XLSX write error: {0}")]
    XlsxWrite(#[from] rust_xlsxwriter::XlsxError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("input '{path}' is {bytes} bytes, exceeding limit of {limit} bytes")]
    FileTooLarge {
        path: String,
        bytes: u64,
        limit: u64,
    },

    // ── Structure ────────────────────────────────────────────────────────────
    #[error("workbook has no sheets")]
    NoSheets,

    #[error("sheet '{sheet}' has no data rows")]
    EmptySheet { sheet: String },

    #[error("sheet '{sheet}' row {row}: {msg}")]
    MalformedRow {
        sheet: String,
        row: usize,
        msg: String,
    },

    #[error("sheet '{sheet}' row {row} col {col}: expected {expected}, got '{actual}'")]
    WrongCellType {
        sheet: String,
        row: usize,
        col: usize,
        expected: &'static str,
        actual: String,
    },

    #[error("sheet '{sheet}' row {row} col {col}: {msg}")]
    ValueOutOfRange {
        sheet: String,
        row: usize,
        col: usize,
        msg: String,
    },

    // ── Validation ───────────────────────────────────────────────────────────
    #[error("validation failed with {count} error(s):\n{messages}")]
    Validation { count: usize, messages: String },

    #[error("sheet '{sheet}' row {row}: {msg}")]
    ValidationRow {
        sheet: String,
        row: usize,
        msg: String,
    },

    // ── Other ────────────────────────────────────────────────────────────────
    #[error("unknown shape kind: '{0}'")]
    UnknownShape(String),

    #[error("edge references unknown label '{label}' in sheet '{sheet}'")]
    UnknownEdgeLabel { sheet: String, label: String },
}
