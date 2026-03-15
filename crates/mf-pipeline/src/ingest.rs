//! Text ingestion — read UTF-8 text from files and directories.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::error::MfError;

/// Read a single UTF-8 text file into a `String`.
pub fn ingest_file(path: &Path) -> Result<String> {
    let text = std::fs::read_to_string(path).map_err(MfError::Io)?;
    Ok(text)
}

/// Concatenate multiple files (joined by `\n\n`).
pub fn ingest_files(paths: &[PathBuf]) -> Result<String> {
    let mut parts: Vec<String> = Vec::with_capacity(paths.len());
    for p in paths {
        parts.push(ingest_file(p)?);
    }
    Ok(parts.join("\n\n"))
}

/// Read all `*.txt` files (non-recursive) in a directory, joined by `\n\n`.
pub fn ingest_directory(dir: &Path) -> Result<String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(MfError::Io)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let p = entry.path();
            if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("txt") {
                Some(p)
            } else {
                None
            }
        })
        .collect();

    // Sort for deterministic ordering.
    paths.sort();
    ingest_files(&paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_txt(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::with_suffix(".txt").unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_ingest_file_basic() {
        let f = temp_txt("hello world");
        let result = ingest_file(f.path()).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_ingest_files_join() {
        let f1 = temp_txt("first");
        let f2 = temp_txt("second");
        let paths = vec![f1.path().to_path_buf(), f2.path().to_path_buf()];
        let result = ingest_files(&paths).unwrap();
        assert_eq!(result, "first\n\nsecond");
    }
}
