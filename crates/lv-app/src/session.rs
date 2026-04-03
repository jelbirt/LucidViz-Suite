//! `session` — save / load a lightweight AppState snapshot.
//!
//! Sessions are persisted as JSON under `~/.lucid-viz/sessions/{name}.json`.
//! The snapshot stores the source-file path, LIS config, slice index, cluster
//! filter bounds, and ego-cluster settings so the user can resume a working
//! state across launches.

// These items are part of the public API surface; they will be used when the
// session management UI is wired up.
#![allow(dead_code)]

use crate::app_state::EgoClusterState;
use anyhow::{Context, Result};
use lv_data::LisConfig;
use lv_gui::state::ExportImageFormat;
use lv_gui::EgoEdgeDirection;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const MAX_SESSION_BYTES: u64 = 16 * 1024 * 1024;

// ── SessionSnapshot ───────────────────────────────────────────────────────────

/// Serialisable snapshot of application state (no GPU handles).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    /// Human-readable name for this session.
    pub name: String,
    /// Path to the source `.lv.xlsx` file (if any).
    pub source_path: Option<PathBuf>,
    /// LIS (animation) configuration.
    pub lis_config: LisConfigSnapshot,
    /// Active animation slice index at save time.
    pub slice_index: u32,
    /// Cluster-value lower bound.
    pub cluster_min: f64,
    /// Cluster-value upper bound.
    pub cluster_max: f64,
    /// Whether ego-cluster mode was active.
    pub ego_mode: bool,
    /// Ego-cluster sub-state.
    pub ego: EgoSnapshot,
    /// Audio panel/runtime settings.
    pub audio: AudioSnapshot,
    /// Export panel settings.
    pub export: ExportSnapshot,
}

/// Serialisable mirror of [`LisConfig`] (no opaque fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LisConfigSnapshot {
    pub lis_value: u32,
    pub target_fps: Option<u32>,
    pub looping: bool,
    pub speed: f32,
}

impl From<&LisConfig> for LisConfigSnapshot {
    fn from(c: &LisConfig) -> Self {
        Self {
            lis_value: c.lis_value,
            target_fps: c.target_fps,
            looping: c.looping,
            speed: c.speed,
        }
    }
}

impl From<LisConfigSnapshot> for LisConfig {
    fn from(s: LisConfigSnapshot) -> Self {
        LisConfig {
            lis_value: s.lis_value,
            target_fps: s.target_fps,
            looping: s.looping,
            speed: s.speed,
        }
    }
}

/// Serialisable mirror of [`EgoClusterState`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgoSnapshot {
    pub selected: Option<String>,
    pub show_secondary: bool,
    pub direction: EgoEdgeDirection,
    pub shared_objects_only: bool,
    pub cluster_value_min: f64,
    pub cluster_value_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSnapshot {
    pub selected_port: String,
    pub live_enabled: bool,
    pub volume: f32,
    pub graduated: bool,
    pub semitone_range: i32,
    pub beats: u32,
    pub hold_slices: u32,
}

impl Default for AudioSnapshot {
    fn default() -> Self {
        Self {
            selected_port: String::new(),
            live_enabled: false,
            volume: 1.0,
            graduated: false,
            semitone_range: 12,
            beats: 1,
            hold_slices: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSnapshot {
    pub output_dir: Option<PathBuf>,
    pub filename_prefix: String,
    pub start_frame: u32,
    pub end_frame: u32,
    pub width: u32,
    pub height: u32,
    pub format: ExportImageFormat,
    pub fps: u32,
    pub crf: u32,
    pub codec: String,
}

impl Default for ExportSnapshot {
    fn default() -> Self {
        Self {
            output_dir: None,
            filename_prefix: "frame".into(),
            start_frame: 0,
            end_frame: 0,
            width: 1920,
            height: 1080,
            format: ExportImageFormat::Png,
            fps: 30,
            crf: 23,
            codec: "libx264".into(),
        }
    }
}

impl From<&EgoClusterState> for EgoSnapshot {
    fn from(e: &EgoClusterState) -> Self {
        Self {
            selected: e.selected.clone(),
            show_secondary: e.show_secondary,
            direction: e.direction,
            shared_objects_only: e.shared_objects_only,
            cluster_value_min: e.cluster_value_min,
            cluster_value_max: e.cluster_value_max,
        }
    }
}

impl From<EgoSnapshot> for EgoClusterState {
    fn from(s: EgoSnapshot) -> Self {
        EgoClusterState {
            selected: s.selected,
            show_secondary: s.show_secondary,
            direction: s.direction,
            shared_objects_only: s.shared_objects_only,
            cluster_value_min: s.cluster_value_min,
            cluster_value_max: s.cluster_value_max,
            cached_ego_edges: None,
            cached_visible: None,
        }
    }
}

// ── persistence helpers ───────────────────────────────────────────────────────

fn sessions_dir() -> Option<PathBuf> {
    dirs_next::home_dir().map(|h| h.join(".lucid-viz").join("sessions"))
}

fn session_path(name: &str) -> Option<PathBuf> {
    // Sanitise name: keep only alphanumeric, dash, underscore
    let safe: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    sessions_dir().map(|d| d.join(format!("{safe}.json")))
}

/// Persist `snapshot` to `~/.lucid-viz/sessions/{name}.json`.
pub fn save_session(snapshot: &SessionSnapshot) -> Result<()> {
    let path = session_path(&snapshot.name).context("cannot determine home directory")?;
    let dir = path.parent().expect("session path has parent");
    std::fs::create_dir_all(dir).with_context(|| format!("create sessions dir {:?}", dir))?;
    let json = serde_json::to_string_pretty(snapshot).context("serialise session")?;
    atomic_write(&path, json.as_bytes()).with_context(|| format!("write session {:?}", path))?;
    Ok(())
}

/// Load a saved session by name.
pub fn load_session(name: &str) -> Result<SessionSnapshot> {
    let path = session_path(name).context("cannot determine home directory")?;
    let bytes = read_bounded_file(&path, MAX_SESSION_BYTES)
        .with_context(|| format!("read session {:?}", path))?;
    let snapshot: SessionSnapshot =
        serde_json::from_slice(&bytes).with_context(|| format!("parse session {:?}", path))?;
    Ok(snapshot)
}

fn read_bounded_file(path: &std::path::Path, limit: u64) -> Result<Vec<u8>> {
    Ok(lv_data::io_util::read_bounded_file(path, limit)?)
}

fn atomic_write(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    Ok(lv_data::io_util::atomic_write(path, bytes)?)
}

/// Delete a saved session by name.
pub fn delete_session(name: &str) -> Result<()> {
    let path = session_path(name).context("cannot determine home directory")?;
    if path.exists() {
        std::fs::remove_file(&path).with_context(|| format!("delete session {:?}", path))?;
    }
    Ok(())
}

/// Rename a saved session.
pub fn rename_session(old_name: &str, new_name: &str) -> Result<()> {
    let old_path = session_path(old_name).context("cannot determine home directory")?;
    let new_path = session_path(new_name).context("cannot determine home directory")?;
    if !old_path.exists() {
        anyhow::bail!("session '{old_name}' does not exist");
    }
    if new_path.exists() {
        anyhow::bail!("session '{new_name}' already exists");
    }
    // Load, update name, save under new path, delete old.
    let bytes = read_bounded_file(&old_path, MAX_SESSION_BYTES)
        .with_context(|| format!("read session {:?}", old_path))?;
    let mut snapshot: SessionSnapshot =
        serde_json::from_slice(&bytes).with_context(|| format!("parse session {:?}", old_path))?;
    snapshot.name = new_name.to_string();
    let json = serde_json::to_string_pretty(&snapshot).context("serialise session")?;
    atomic_write(&new_path, json.as_bytes())
        .with_context(|| format!("write session {:?}", new_path))?;
    std::fs::remove_file(&old_path)
        .with_context(|| format!("remove old session {:?}", old_path))?;
    Ok(())
}

/// List all saved session names (stem of each `*.json` in the sessions dir).
pub fn list_sessions() -> Vec<String> {
    let Some(dir) = sessions_dir() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
        .filter_map(|e| {
            e.path()
                .file_stem()
                .and_then(|s| s.to_str())
                .map(String::from)
        })
        .collect()
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn with_tmp_home(f: impl FnOnce()) {
        // We can't easily override dirs_next::home_dir, so we test serialisation
        // directly instead.
        let _ = f;
    }

    #[test]
    fn snapshot_round_trip() {
        let snap = SessionSnapshot {
            name: "test_session".to_string(),
            source_path: Some(PathBuf::from("/tmp/test.xlsx")),
            lis_config: LisConfigSnapshot {
                lis_value: 60,
                target_fps: None,
                looping: true,
                speed: 1.0,
            },
            slice_index: 7,
            cluster_min: 0.0,
            cluster_max: 5.0,
            ego_mode: true,
            ego: EgoSnapshot {
                selected: Some("node_0".to_string()),
                show_secondary: false,
                direction: EgoEdgeDirection::Both,
                shared_objects_only: true,
                cluster_value_min: 0.0,
                cluster_value_max: 5.0,
            },
            audio: AudioSnapshot {
                selected_port: "Port A".into(),
                live_enabled: true,
                volume: 0.8,
                graduated: true,
                semitone_range: 7,
                beats: 3,
                hold_slices: 4,
            },
            export: ExportSnapshot {
                output_dir: Some(PathBuf::from("/tmp/exports")),
                filename_prefix: "demo".into(),
                start_frame: 2,
                end_frame: 8,
                width: 1280,
                height: 720,
                format: ExportImageFormat::Tga,
                fps: 24,
                crf: 20,
                codec: "libx265".into(),
            },
        };

        let json = serde_json::to_string(&snap).unwrap();
        let snap2: SessionSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snap2.name, "test_session");
        assert_eq!(snap2.slice_index, 7);
        assert_eq!(snap2.ego.selected, Some("node_0".to_string()));
        assert_eq!(snap2.ego.direction, EgoEdgeDirection::Both);
        assert_eq!(snap2.audio.selected_port, "Port A");
        assert_eq!(snap2.export.filename_prefix, "demo");
    }

    #[test]
    fn lis_config_roundtrip() {
        let lc = LisConfig {
            lis_value: 120,
            target_fps: Some(60),
            looping: false,
            speed: 2.0,
        };
        let snap = LisConfigSnapshot::from(&lc);
        let lc2 = LisConfig::from(snap);
        assert_eq!(lc2.lis_value, 120);
        assert_eq!(lc2.target_fps, Some(60));
        assert!(!lc2.looping);
    }

    #[test]
    fn save_load_session_tempdir() {
        let dir = TempDir::new().unwrap();
        let snap = SessionSnapshot {
            name: "demo".to_string(),
            source_path: None,
            lis_config: LisConfigSnapshot {
                lis_value: 30,
                target_fps: None,
                looping: true,
                speed: 1.0,
            },
            slice_index: 0,
            cluster_min: 0.0,
            cluster_max: 100.0,
            ego_mode: false,
            ego: EgoSnapshot {
                selected: None,
                show_secondary: false,
                direction: EgoEdgeDirection::Both,
                shared_objects_only: false,
                cluster_value_min: 0.0,
                cluster_value_max: 100.0,
            },
            audio: AudioSnapshot::default(),
            export: ExportSnapshot::default(),
        };

        let path = dir.path().join("demo.json");
        let json = serde_json::to_string_pretty(&snap).unwrap();
        std::fs::write(&path, &json).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let snap2: SessionSnapshot = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(snap2.name, "demo");
        assert_eq!(snap2.lis_config.lis_value, 30);
    }

    #[test]
    fn bounded_session_read_rejects_oversized_files() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("oversized.json");
        std::fs::write(&path, vec![b'x'; 32]).unwrap();
        let err = read_bounded_file(&path, 8).expect_err("oversized session should fail");
        assert!(format!("{err:#}").contains("exceeding limit"));
    }
}
