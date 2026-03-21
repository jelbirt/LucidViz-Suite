//! Shared application state threaded through the GUI layer.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;

use lv_data::{EtvDataset, LisBuffer, LisConfig, ShapeKind};
use mf_pipeline::types::{MfOutput, MfSeriesOutput};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AsInputSource {
    #[default]
    Dataset,
    MatrixForge,
    MatrixForgeSeries,
}

/// Per-object visual override applied when building LisFrames.
#[derive(Clone, Debug)]
pub struct ObjectOverride {
    pub shape: Option<ShapeKind>,
    pub color: Option<[f32; 3]>,
    pub size: Option<f64>,
}

/// Playback state for the LIS transport.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlayState {
    #[default]
    Playing,
    Paused,
    Stopped,
}

/// The full mutable application state shared between the GUI and renderer.
pub struct AppState {
    // ── Dataset ─────────────────────────────────────────────────────────────
    pub dataset: Option<EtvDataset>,
    pub mf_output: Option<MfOutput>,
    pub mf_series_output: Option<MfSeriesOutput>,
    pub source_path: Option<PathBuf>,
    pub as_input_source: AsInputSource,

    // ── LIS ─────────────────────────────────────────────────────────────────
    pub lis_config: LisConfig,
    pub lis_buffer: Option<LisBuffer>,
    pub slice_index: u32,
    pub play_state: PlayState,

    // ── Overrides ────────────────────────────────────────────────────────────
    pub overrides: HashMap<String, ObjectOverride>,

    // ── Cluster filter ───────────────────────────────────────────────────────
    pub cluster_min: f64,
    pub cluster_max: f64,
    pub ego_mode: bool,
    pub secondary_edges: bool,
    pub shared_only: bool,

    // ── Dirty flags ─────────────────────────────────────────────────────────
    /// Set to true when the LIS buffer must be rebuilt (LIS value changed etc.)
    pub rebuild_lis: bool,
    /// Set to true when a new dataset was just loaded.
    pub dataset_changed: bool,

    // ── File loader status ───────────────────────────────────────────────────
    pub load_error: Option<String>,

    // ── Pipeline progress receivers ─────────────────────────────────────────
    pub as_job: Option<PipelineJob>,
    pub mf_job: Option<PipelineJob>,
}

/// A running background pipeline task.
pub struct PipelineJob {
    pub receiver: mpsc::Receiver<PipelineEvent>,
    pub last_step: String,
    pub last_pct: f32,
    pub result: Option<Result<PathBuf, String>>,
}

/// Events emitted by background pipeline threads.
#[derive(Debug)]
pub enum PipelineEvent {
    Progress { step: String, pct: f32 },
    Done(Result<PathBuf, String>),
}

impl AppState {
    pub fn new() -> Self {
        Self {
            dataset: None,
            mf_output: None,
            mf_series_output: None,
            source_path: None,
            as_input_source: AsInputSource::Dataset,
            lis_config: LisConfig::default(),
            lis_buffer: None,
            slice_index: 0,
            play_state: PlayState::Playing,
            overrides: HashMap::new(),
            cluster_min: f64::NEG_INFINITY,
            cluster_max: f64::INFINITY,
            ego_mode: false,
            secondary_edges: false,
            shared_only: false,
            rebuild_lis: false,
            dataset_changed: false,
            load_error: None,
            as_job: None,
            mf_job: None,
        }
    }

    /// Drain progress events from any running background jobs.
    pub fn poll_jobs(&mut self) {
        for job in [&mut self.as_job, &mut self.mf_job].into_iter().flatten() {
            while let Ok(ev) = job.receiver.try_recv() {
                match ev {
                    PipelineEvent::Progress { step, pct } => {
                        job.last_step = step;
                        job.last_pct = pct;
                    }
                    PipelineEvent::Done(r) => {
                        job.result = Some(r);
                    }
                }
            }
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
