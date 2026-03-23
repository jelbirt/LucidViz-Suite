//! Shared application state threaded through the GUI layer.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{atomic::AtomicBool, Arc};

use lv_data::{EtvDataset, LisBuffer, LisConfig, ShapeKind};
use mf_pipeline::types::{MfOutput, MfSeriesOutput};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct AsRunOutcome {
    pub dataset: EtvDataset,
    pub dataset_path: PathBuf,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PendingDatasetLoad {
    pub dataset: EtvDataset,
    pub source_path: PathBuf,
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeSnapshot {
    pub dataset: Option<EtvDataset>,
    pub source_path: Option<PathBuf>,
    pub lis_buffer: Option<LisBuffer>,
    pub slice_index: u32,
}

#[derive(Clone, Debug)]
pub enum PipelineOutcome {
    AsRun(AsRunOutcome),
    File(PathBuf),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportImageFormat {
    Png,
    Tga,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportKind {
    CurrentFrame,
    Sequence,
    Video,
}

#[derive(Clone, Debug)]
pub struct ExportRequest {
    pub kind: ExportKind,
    pub output_dir: PathBuf,
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

pub struct ExportJob {
    pub kind: ExportKind,
    pub progress: f32,
    pub progress_rx: mpsc::Receiver<f32>,
    pub result_rx: mpsc::Receiver<Result<String, String>>,
    pub cancel_flag: Arc<AtomicBool>,
}

#[derive(Clone, Debug)]
pub enum AudioRequest {
    RefreshPorts,
    Connect(String),
    Disconnect,
    TestTone,
}

#[derive(Clone, Debug)]
pub enum SessionRequest {
    RefreshList,
    Save(String),
    Load(String),
}

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

/// Directional interpretation for ego-cluster expansion and edge display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EgoEdgeDirection {
    Incoming,
    Outgoing,
    #[default]
    Both,
}

/// The full mutable application state shared between the GUI and renderer.
pub struct AppState {
    // ── Runtime snapshot (renderer-owned, GUI-read-only) ───────────────────
    runtime: RuntimeSnapshot,

    // ── Dataset / pipeline inputs ───────────────────────────────────────────
    pub mf_output: Option<MfOutput>,
    pub mf_series_output: Option<MfSeriesOutput>,
    pub pending_dataset_load: Option<PendingDatasetLoad>,
    pub as_input_source: AsInputSource,

    // ── LIS ─────────────────────────────────────────────────────────────────
    pub lis_config: LisConfig,
    pub pending_slice_index: Option<u32>,
    pub play_state: PlayState,

    // ── Runtime integration requests ────────────────────────────────────────
    pub pending_export_request: Option<ExportRequest>,
    pub export_job: Option<ExportJob>,
    pub export_output_dir: Option<PathBuf>,
    pub export_filename_prefix: String,
    pub export_start_frame: u32,
    pub export_end_frame: u32,
    pub export_width: u32,
    pub export_height: u32,
    pub export_format: ExportImageFormat,
    pub export_fps: u32,
    pub export_crf: u32,
    pub export_codec: String,
    pub export_status: Option<String>,
    pub pending_audio_request: Option<AudioRequest>,
    pub audio_selected_port: String,
    pub audio_ports: Vec<String>,
    pub audio_connected: bool,
    pub audio_live_enabled: bool,
    pub audio_volume: f32,
    pub audio_graduated: bool,
    pub audio_semitone_range: i32,
    pub audio_beats: u32,
    pub audio_hold_slices: u32,
    pub audio_status: Option<String>,
    pub pending_session_request: Option<SessionRequest>,
    pub session_name: String,
    pub saved_sessions: Vec<String>,
    pub session_status: Option<String>,

    // ── Overrides ────────────────────────────────────────────────────────────
    pub overrides: HashMap<String, ObjectOverride>,

    // ── Cluster filter ───────────────────────────────────────────────────────
    pub cluster_min: f64,
    pub cluster_max: f64,
    pub ego_mode: bool,
    pub ego_direction: EgoEdgeDirection,
    pub secondary_edges: bool,
    pub shared_only: bool,

    // ── Dirty flags ─────────────────────────────────────────────────────────
    /// Set to true when the LIS buffer must be rebuilt (LIS value changed etc.)
    pub rebuild_lis: bool,

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
    pub result: Option<Result<PipelineOutcome, String>>,
}

/// Events emitted by background pipeline threads.
#[derive(Debug)]
pub enum PipelineEvent {
    Progress { step: String, pct: f32 },
    Done(Result<PipelineOutcome, String>),
}

impl AppState {
    pub fn new() -> Self {
        Self {
            runtime: RuntimeSnapshot::default(),
            mf_output: None,
            mf_series_output: None,
            pending_dataset_load: None,
            as_input_source: AsInputSource::Dataset,
            lis_config: LisConfig::default(),
            pending_slice_index: None,
            play_state: PlayState::Playing,
            pending_export_request: None,
            export_job: None,
            export_output_dir: None,
            export_filename_prefix: "frame".into(),
            export_start_frame: 0,
            export_end_frame: 0,
            export_width: 1920,
            export_height: 1080,
            export_format: ExportImageFormat::Png,
            export_fps: 30,
            export_crf: 23,
            export_codec: "libx264".into(),
            export_status: None,
            pending_audio_request: None,
            audio_selected_port: String::new(),
            audio_ports: Vec::new(),
            audio_connected: false,
            audio_live_enabled: false,
            audio_volume: 1.0,
            audio_graduated: false,
            audio_semitone_range: 12,
            audio_beats: 1,
            audio_hold_slices: 2,
            audio_status: None,
            pending_session_request: None,
            session_name: String::new(),
            saved_sessions: Vec::new(),
            session_status: None,
            overrides: HashMap::new(),
            cluster_min: f64::NEG_INFINITY,
            cluster_max: f64::INFINITY,
            ego_mode: false,
            ego_direction: EgoEdgeDirection::Both,
            secondary_edges: false,
            shared_only: false,
            rebuild_lis: false,
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

    pub fn runtime(&self) -> &RuntimeSnapshot {
        &self.runtime
    }

    pub fn dataset(&self) -> Option<&EtvDataset> {
        self.runtime.dataset.as_ref()
    }

    pub fn source_path(&self) -> Option<&PathBuf> {
        self.runtime.source_path.as_ref()
    }

    pub fn lis_buffer(&self) -> Option<&LisBuffer> {
        self.runtime.lis_buffer.as_ref()
    }

    pub fn slice_index(&self) -> u32 {
        self.runtime.slice_index
    }

    pub fn sync_runtime_snapshot(
        &mut self,
        dataset: &EtvDataset,
        source_path: Option<PathBuf>,
        lis_buffer: &LisBuffer,
        slice_index: u32,
    ) {
        self.runtime.dataset = Some(dataset.clone());
        self.runtime.source_path = source_path;
        self.runtime.lis_buffer = Some(lis_buffer.clone());
        self.runtime.slice_index = slice_index;
    }

    pub fn queue_dataset_load(&mut self, dataset: EtvDataset, source_path: PathBuf) {
        self.pending_dataset_load = Some(PendingDatasetLoad {
            dataset,
            source_path,
        });
        self.as_input_source = AsInputSource::Dataset;
        self.mf_output = None;
        self.mf_series_output = None;
        self.pending_slice_index = Some(0);
        self.load_error = None;
    }

    pub fn poll_export_job(&mut self) {
        let mut finished = false;
        if let Some(job) = self.export_job.as_mut() {
            while let Ok(progress) = job.progress_rx.try_recv() {
                job.progress = progress;
            }
            while let Ok(result) = job.result_rx.try_recv() {
                self.export_status = Some(match result {
                    Ok(msg) => msg,
                    Err(err) => format!("Export failed: {err}"),
                });
                finished = true;
            }
        }
        if finished {
            self.export_job = None;
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::AppState;
    use lv_data::{EtvDataset, LisBuffer, LisConfig};
    use std::path::PathBuf;

    fn sample_dataset() -> EtvDataset {
        EtvDataset {
            source_path: None,
            sheets: vec![],
            all_labels: vec!["alpha".into()],
        }
    }

    #[test]
    fn runtime_snapshot_updates_through_sync_helper() {
        let mut state = AppState::new();
        let dataset = sample_dataset();
        let lis_buffer = LisBuffer {
            frames: vec![],
            lis: 30,
            total_frames: 12,
            streaming: false,
        };
        let path = Some(PathBuf::from("/tmp/sample.json"));

        state.sync_runtime_snapshot(&dataset, path.clone(), &lis_buffer, 7);

        assert_eq!(
            state.dataset().map(|d| d.all_labels.clone()),
            Some(dataset.all_labels)
        );
        assert_eq!(state.source_path().cloned(), path);
        assert_eq!(state.lis_buffer().map(|b| b.total_frames), Some(12));
        assert_eq!(state.slice_index(), 7);
    }

    #[test]
    fn queue_dataset_load_keeps_runtime_snapshot_authoritative() {
        let mut state = AppState::new();
        let runtime_dataset = sample_dataset();
        let lis_buffer = LisBuffer {
            frames: vec![],
            lis: LisConfig::default().lis_value,
            total_frames: 3,
            streaming: false,
        };
        state.sync_runtime_snapshot(
            &runtime_dataset,
            Some(PathBuf::from("/tmp/runtime.json")),
            &lis_buffer,
            2,
        );

        let pending_dataset = EtvDataset {
            source_path: None,
            sheets: vec![],
            all_labels: vec!["beta".into()],
        };
        state.queue_dataset_load(pending_dataset, PathBuf::from("/tmp/pending.json"));

        assert_eq!(
            state.dataset().unwrap().all_labels,
            vec!["alpha".to_string()]
        );
        assert_eq!(state.slice_index(), 2);
        assert!(state.pending_dataset_load.is_some());
        assert_eq!(state.pending_slice_index, Some(0));
    }
}
