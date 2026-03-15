//! `prefs` — UserPreferences: load/save from `~/.lucid-viz/prefs.json`.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── UserPreferences ────────────────────────────────────────────────────────────

/// Persistent user preferences written to `~/.lucid-viz/prefs.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UserPreferences {
    /// Most-recently-opened files (capped at 10).
    pub recent_files: Vec<PathBuf>,
    /// Last window width in logical pixels.
    pub window_width: u32,
    /// Last window height in logical pixels.
    pub window_height: u32,
    /// Default LIS (animation steps per cycle).
    pub default_lis: u32,
    /// Default export FPS (`None` = use application default).
    pub default_fps: Option<u32>,
    /// MIDI output port name (`None` = first available).
    pub audio_port: Option<String>,
    /// Worker threads for betweenness centrality (`None` = num_cpus - 1).
    pub brandes_threads: Option<usize>,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            recent_files: Vec::new(),
            window_width: 1280,
            window_height: 720,
            default_lis: 60,
            default_fps: None,
            audio_port: None,
            brandes_threads: None,
        }
    }
}

impl UserPreferences {
    /// Return the path `~/.lucid-viz/prefs.json`, or `None` if home dir is unknown.
    fn prefs_path() -> Option<PathBuf> {
        dirs_next::home_dir().map(|h| h.join(".lucid-viz").join("prefs.json"))
    }

    /// Ensure `~/.lucid-viz/` exists and return the prefs path.
    fn ensure_dir() -> Result<PathBuf> {
        let path = Self::prefs_path().context("cannot determine home directory")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create_dir_all {:?}", parent))?;
        }
        Ok(path)
    }

    /// Load preferences from disk.  Returns [`Default`] if the file is absent.
    pub fn load() -> Self {
        match Self::try_load() {
            Ok(p) => p,
            Err(e) => {
                log::warn!("Failed to load prefs: {e:#}; using defaults");
                Self::default()
            }
        }
    }

    fn try_load() -> Result<Self> {
        let path = Self::prefs_path().context("cannot determine home directory")?;
        let bytes = std::fs::read(&path).with_context(|| format!("read {:?}", path))?;
        let prefs: Self =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {:?}", path))?;
        Ok(prefs)
    }

    /// Persist preferences to `~/.lucid-viz/prefs.json`.
    pub fn save(&self) -> Result<()> {
        let path = Self::ensure_dir()?;
        let json = serde_json::to_string_pretty(self).context("serialise preferences")?;
        std::fs::write(&path, json).with_context(|| format!("write {:?}", path))?;
        Ok(())
    }

    /// Add `path` to the recent-files list (dedup, capped at 10).
    #[allow(dead_code)]
    pub fn push_recent(&mut self, path: PathBuf) {
        self.recent_files.retain(|p| p != &path);
        self.recent_files.insert(0, path);
        self.recent_files.truncate(10);
    }
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_prefs_round_trip() {
        let p = UserPreferences::default();
        let json = serde_json::to_string(&p).unwrap();
        let p2: UserPreferences = serde_json::from_str(&json).unwrap();
        assert_eq!(p.window_width, p2.window_width);
        assert_eq!(p.window_height, p2.window_height);
        assert_eq!(p.default_lis, p2.default_lis);
    }

    #[test]
    fn push_recent_caps_and_dedupes() {
        let mut p = UserPreferences::default();
        for i in 0..12 {
            p.push_recent(PathBuf::from(format!("/tmp/file{i}.xlsx")));
        }
        assert_eq!(p.recent_files.len(), 10, "capped at 10");
        // Pushing an existing entry moves it to front without growing list
        let top = p.recent_files[0].clone();
        p.push_recent(p.recent_files[5].clone());
        assert_eq!(p.recent_files.len(), 10, "still capped after dedup");
        assert_ne!(p.recent_files[0], top, "moved to front");
    }
}
