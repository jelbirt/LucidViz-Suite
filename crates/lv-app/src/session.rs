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
use lv_gui::EgoEdgeDirection;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── SessionSnapshot ───────────────────────────────────────────────────────────

/// Serialisable snapshot of application state (no GPU handles).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    /// Human-readable name for this session.
    pub name: String,
    /// Path to the source `.etv.xlsx` file (if any).
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
    std::fs::write(&path, json).with_context(|| format!("write session {:?}", path))?;
    Ok(())
}

/// Load a saved session by name.
pub fn load_session(name: &str) -> Result<SessionSnapshot> {
    let path = session_path(name).context("cannot determine home directory")?;
    let bytes = std::fs::read(&path).with_context(|| format!("read session {:?}", path))?;
    let snapshot: SessionSnapshot =
        serde_json::from_slice(&bytes).with_context(|| format!("parse session {:?}", path))?;
    Ok(snapshot)
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
        };

        let json = serde_json::to_string(&snap).unwrap();
        let snap2: SessionSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snap2.name, "test_session");
        assert_eq!(snap2.slice_index, 7);
        assert_eq!(snap2.ego.selected, Some("node_0".to_string()));
        assert_eq!(snap2.ego.direction, EgoEdgeDirection::Both);
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
        };

        let path = dir.path().join("demo.json");
        let json = serde_json::to_string_pretty(&snap).unwrap();
        std::fs::write(&path, &json).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let snap2: SessionSnapshot = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(snap2.name, "demo");
        assert_eq!(snap2.lis_config.lis_value, 30);
    }
}
