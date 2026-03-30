//! Shared file I/O utilities: atomic writes and bounded reads.
//!
//! These helpers are used across multiple crates to ensure safe,
//! crash-consistent file output and size-bounded file reading.

use std::path::{Path, PathBuf};

/// Generate a temporary sibling path for atomic write operations.
///
/// Returns a path like `.{filename}.tmp-{pid}` in the same directory.
pub fn temp_path(path: &Path) -> std::io::Result<PathBuf> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "path must have a file name",
            )
        })?;
    Ok(path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id())))
}

/// Atomically rename `tmp_path` to `dst_path`, with a fallback
/// delete-then-rename for platforms where rename over an existing file fails.
///
/// On POSIX, `rename` atomically replaces the destination, so the fallback
/// path is only reached on Windows where rename fails if the destination
/// exists.  The fallback creates a brief window where neither file exists;
/// the tmp file is preserved on any failure so data is not lost.
pub fn replace_file(tmp_path: &Path, dst_path: &Path) -> std::io::Result<()> {
    match std::fs::rename(tmp_path, dst_path) {
        Ok(()) => Ok(()),
        Err(_) if dst_path.exists() => {
            // Destination exists and rename failed (Windows).
            // Back up the destination so we can restore it on failure.
            let backup = dst_path.with_extension("bak");
            std::fs::rename(dst_path, &backup)?;
            if let Err(rename_err) = std::fs::rename(tmp_path, dst_path) {
                // Restore the original file from backup.
                let _ = std::fs::rename(&backup, dst_path);
                Err(rename_err)
            } else {
                // Success — remove the backup.
                let _ = std::fs::remove_file(&backup);
                Ok(())
            }
        }
        Err(err) => Err(err),
    }
}

/// Write `bytes` to `path` atomically via a temp file + rename.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = temp_path(path)?;
    std::fs::write(&tmp, bytes)?;
    replace_file(&tmp, path)
}

/// Read a file into memory, returning an error if it exceeds `limit` bytes.
pub fn read_bounded_file(path: &Path, limit: u64) -> std::io::Result<Vec<u8>> {
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > limit {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "file '{}' is {} bytes, exceeding limit of {} bytes",
                path.display(),
                metadata.len(),
                limit
            ),
        ));
    }
    std::fs::read(path)
}

/// Save a `rust_xlsxwriter::Workbook` atomically via temp path + rename.
pub fn save_workbook_atomic(
    wb: &mut rust_xlsxwriter::Workbook,
    path: &Path,
) -> Result<(), rust_xlsxwriter::XlsxError> {
    let tmp = temp_path(path).map_err(rust_xlsxwriter::XlsxError::IoError)?;
    wb.save(&tmp)?;
    replace_file(&tmp, path).map_err(rust_xlsxwriter::XlsxError::IoError)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        atomic_write(&path, b"hello").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    #[test]
    fn atomic_write_overwrites_existing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "old").unwrap();
        atomic_write(&path, b"new").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
    }

    #[test]
    fn read_bounded_file_rejects_oversized() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.txt");
        std::fs::write(&path, vec![b'x'; 100]).unwrap();
        let err = read_bounded_file(&path, 50).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn read_bounded_file_accepts_within_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ok.txt");
        std::fs::write(&path, b"data").unwrap();
        let bytes = read_bounded_file(&path, 1024).unwrap();
        assert_eq!(bytes, b"data");
    }

    #[test]
    fn temp_path_rejects_empty_path() {
        let err = temp_path(Path::new("")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
