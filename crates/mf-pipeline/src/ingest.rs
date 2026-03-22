//! Text ingestion — read UTF-8 text from files and directories.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::error::MfError;

const MAX_TEXT_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct TextSource {
    pub label: String,
    pub path: PathBuf,
    pub text: String,
}

/// Read a single UTF-8 text file into a `String`.
pub fn ingest_file(path: &Path) -> Result<String> {
    let text = read_text_file_with_limit(path, MAX_TEXT_BYTES)?;
    Ok(text)
}

fn read_text_file_with_limit(path: &Path, limit: u64) -> Result<String> {
    let metadata = std::fs::metadata(path).map_err(MfError::Io)?;
    if metadata.len() > limit {
        return Err(MfError::FileTooLarge {
            path: path.display().to_string(),
            bytes: metadata.len(),
            limit,
        }
        .into());
    }
    Ok(std::fs::read_to_string(path).map_err(MfError::Io)?)
}

/// Concatenate multiple files (joined by `\n\n`).
pub fn ingest_files(paths: &[PathBuf]) -> Result<String> {
    let mut parts: Vec<String> = Vec::with_capacity(paths.len());
    for p in paths {
        parts.push(ingest_file(p)?);
    }
    Ok(parts.join("\n\n"))
}

/// Read a mix of files and directories, joining all discovered text with `\n\n`.
pub fn ingest_inputs(paths: &[PathBuf]) -> Result<String> {
    let mut parts: Vec<String> = Vec::new();
    for path in paths {
        if path.is_dir() {
            let text = ingest_directory(path)?;
            if !text.is_empty() {
                parts.push(text);
            }
        } else {
            parts.push(ingest_file(path)?);
        }
    }
    Ok(parts.join("\n\n"))
}

/// Discover individual text sources from a mix of files and directories.
pub fn discover_text_sources(paths: &[PathBuf]) -> Result<Vec<TextSource>> {
    let mut sources = Vec::new();
    for path in paths {
        if path.is_dir() {
            for file_path in list_directory_text_files(path)? {
                let text = ingest_file(&file_path)?;
                sources.push(TextSource {
                    label: file_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("slice")
                        .to_string(),
                    path: file_path,
                    text,
                });
            }
        } else {
            sources.push(TextSource {
                label: path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("slice")
                    .to_string(),
                path: path.clone(),
                text: ingest_file(path)?,
            });
        }
    }
    Ok(sources)
}

/// Read all `*.txt` files (non-recursive) in a directory, joined by `\n\n`.
pub fn ingest_directory(dir: &Path) -> Result<String> {
    let paths = list_directory_text_files(dir)?;
    ingest_files(&paths)
}

fn list_directory_text_files(dir: &Path) -> Result<Vec<PathBuf>> {
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
    Ok(paths)
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

    #[test]
    fn test_ingest_inputs_mixed_file_and_directory() {
        let file = temp_txt("standalone");
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "inside dir").unwrap();
        std::fs::write(dir.path().join("ignored.md"), "ignored").unwrap();

        let result = ingest_inputs(&[file.path().to_path_buf(), dir.path().to_path_buf()]).unwrap();
        assert_eq!(result, "standalone\n\ninside dir");
    }

    #[test]
    fn test_discover_text_sources_expands_directories() {
        let file = temp_txt("standalone");
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "inside dir").unwrap();
        std::fs::write(dir.path().join("b.txt"), "inside dir two").unwrap();

        let sources =
            discover_text_sources(&[file.path().to_path_buf(), dir.path().to_path_buf()]).unwrap();
        assert_eq!(sources.len(), 3);
        assert_eq!(sources[0].text, "standalone");
        assert_eq!(sources[1].label, "a");
        assert_eq!(sources[2].label, "b");
    }

    #[test]
    fn test_ingest_file_rejects_oversized_input() {
        let f = temp_txt("abcdef");
        let err = read_text_file_with_limit(f.path(), 3).expect_err("oversized text should fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("exceeding limit"));
    }
}
