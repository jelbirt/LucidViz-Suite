//! Shared application state threaded through the GUI layer.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{atomic::AtomicBool, Arc};

use lv_data::{LisBuffer, LisConfig, LvDataset, ShapeKind};
use mf_pipeline::types::{MfOutput, MfSeriesOutput};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct AsRunOutcome {
    pub dataset: LvDataset,
    pub dataset_path: PathBuf,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PendingDatasetLoad {
    pub dataset: LvDataset,
    pub source_path: PathBuf,
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeSnapshot {
    pub dataset: Option<LvDataset>,
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

/// Sonification mapping preset — determines how data dimensions map to MIDI parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SonificationMapping {
    /// Centrality magnitude → pitch (default graduated behavior).
    #[default]
    CentralityToPitch,
    /// Degree centrality → velocity (louder nodes have more connections).
    DegreeToVelocity,
    /// Betweenness centrality → pitch, closeness → velocity.
    BetweennessPitchClosenessVelocity,
    /// Cluster membership index → MIDI channel (up to 16 clusters).
    ClusterToChannel,
}

impl SonificationMapping {
    pub const ALL: &'static [SonificationMapping] = &[
        SonificationMapping::CentralityToPitch,
        SonificationMapping::DegreeToVelocity,
        SonificationMapping::BetweennessPitchClosenessVelocity,
        SonificationMapping::ClusterToChannel,
    ];
}

impl std::fmt::Display for SonificationMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SonificationMapping::CentralityToPitch => write!(f, "Centrality → Pitch"),
            SonificationMapping::DegreeToVelocity => write!(f, "Degree → Velocity"),
            SonificationMapping::BetweennessPitchClosenessVelocity => {
                write!(f, "Betweenness → Pitch, Closeness → Velocity")
            }
            SonificationMapping::ClusterToChannel => write!(f, "Cluster → Channel"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum SessionRequest {
    RefreshList,
    Save(String),
    Load(String),
    Delete(String),
    Rename { from: String, to: String },
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

/// Grouped export-related state.
pub struct ExportState {
    pub pending_request: Option<ExportRequest>,
    pub job: Option<ExportJob>,
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
    pub status: Option<String>,
}

impl Default for ExportState {
    fn default() -> Self {
        Self {
            pending_request: None,
            job: None,
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
            status: None,
        }
    }
}

/// Grouped audio-related state.
pub struct AudioState {
    pub pending_request: Option<AudioRequest>,
    pub selected_port: String,
    pub ports: Vec<String>,
    pub connected: bool,
    pub live_enabled: bool,
    pub volume: f32,
    pub graduated: bool,
    pub semitone_range: i32,
    pub beats: u32,
    pub hold_slices: u32,
    pub mapping: SonificationMapping,
    pub status: Option<String>,
}

impl Default for AudioState {
    fn default() -> Self {
        Self {
            pending_request: None,
            selected_port: String::new(),
            ports: Vec::new(),
            connected: false,
            live_enabled: false,
            volume: 1.0,
            graduated: false,
            semitone_range: 12,
            beats: 1,
            hold_slices: 2,
            mapping: SonificationMapping::default(),
            status: None,
        }
    }
}

/// Grouped session-related state.
#[derive(Default)]
pub struct SessionState {
    pub pending_request: Option<SessionRequest>,
    pub name: String,
    pub saved_sessions: Vec<String>,
    pub status: Option<String>,
    pub loading: bool,
    pub confirm_delete: Option<String>,
    pub renaming: Option<String>,
    pub rename_buffer: String,
}

/// Grouped cluster-filter state.
pub struct ClusterState {
    pub min: f64,
    pub max: f64,
    pub ego_mode: bool,
    pub ego_direction: EgoEdgeDirection,
    pub secondary_edges: bool,
    pub shared_only: bool,
    /// Cached data range — recomputed only when dataset changes.
    pub cached_data_range: Option<(f64, f64)>,
}

impl Default for ClusterState {
    fn default() -> Self {
        Self {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
            ego_mode: false,
            ego_direction: EgoEdgeDirection::Both,
            secondary_edges: false,
            shared_only: false,
            cached_data_range: None,
        }
    }
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

    // ── Grouped sub-states ──────────────────────────────────────────────────
    pub export: ExportState,
    pub audio: AudioState,
    pub session: SessionState,
    pub cluster: ClusterState,

    // ── Overrides ────────────────────────────────────────────────────────────
    pub overrides: HashMap<String, ObjectOverride>,

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
            export: ExportState::default(),
            audio: AudioState::default(),
            session: SessionState::default(),
            cluster: ClusterState::default(),
            overrides: HashMap::new(),
            rebuild_lis: false,
            load_error: None,
            as_job: None,
            mf_job: None,
        }
    }

    /// Drain progress events from any running background jobs.
    pub fn poll_jobs(&mut self) {
        for job in [&mut self.as_job, &mut self.mf_job].into_iter().flatten() {
            // Skip polling completed jobs — nothing left to drain.
            if job.result.is_some() {
                continue;
            }
            while let Ok(ev) = job.receiver.try_recv() {
                match ev {
                    PipelineEvent::Progress { step, pct } => {
                        job.last_step = step;
                        job.last_pct = pct;
                    }
                    PipelineEvent::Done(r) => {
                        job.result = Some(r);
                        break; // No more useful events after Done.
                    }
                }
            }
        }
    }

    /// Clear a completed AS pipeline job. Call after the GUI has consumed the result.
    pub fn clear_completed_as_job(&mut self) {
        if self.as_job.as_ref().is_some_and(|j| j.result.is_some()) {
            self.as_job = None;
        }
    }

    /// Clear a completed MF pipeline job. Call after the GUI has consumed the result.
    pub fn clear_completed_mf_job(&mut self) {
        if self.mf_job.as_ref().is_some_and(|j| j.result.is_some()) {
            self.mf_job = None;
        }
    }

    pub fn runtime(&self) -> &RuntimeSnapshot {
        &self.runtime
    }

    pub fn dataset(&self) -> Option<&LvDataset> {
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
        dataset: &LvDataset,
        source_path: Option<PathBuf>,
        lis_buffer: &LisBuffer,
        slice_index: u32,
    ) {
        self.runtime.dataset = Some(dataset.clone());
        self.runtime.source_path = source_path;
        self.runtime.lis_buffer = Some(lis_buffer.clone());
        self.runtime.slice_index = slice_index;
    }

    pub fn queue_dataset_load(&mut self, dataset: LvDataset, source_path: PathBuf) {
        self.pending_dataset_load = Some(PendingDatasetLoad {
            dataset,
            source_path,
        });
        self.as_input_source = AsInputSource::Dataset;
        self.cluster.cached_data_range = None;
        self.mf_output = None;
        self.mf_series_output = None;
        self.pending_slice_index = Some(0);
        self.load_error = None;
    }

    pub fn poll_export_job(&mut self) {
        let mut finished = false;
        if let Some(job) = self.export.job.as_mut() {
            while let Ok(progress) = job.progress_rx.try_recv() {
                job.progress = progress;
            }
            while let Ok(result) = job.result_rx.try_recv() {
                self.export.status = Some(match result {
                    Ok(msg) => msg,
                    Err(err) => format!("Export failed: {err}"),
                });
                finished = true;
            }
        }
        if finished {
            self.export.job = None;
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
    use lv_data::{LisBuffer, LisConfig, LvDataset};
    use std::path::PathBuf;

    fn sample_dataset() -> LvDataset {
        LvDataset {
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

        let pending_dataset = LvDataset {
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

    #[test]
    fn export_state_defaults_are_sane() {
        use super::ExportState;
        let export = ExportState::default();
        assert_eq!(export.width, 1920);
        assert_eq!(export.height, 1080);
        assert_eq!(export.fps, 30);
        assert_eq!(export.crf, 23);
        assert_eq!(export.codec, "libx264");
        assert!(export.job.is_none());
        assert!(export.pending_request.is_none());
    }

    #[test]
    fn audio_state_defaults_are_sane() {
        use super::AudioState;
        let audio = AudioState::default();
        assert_eq!(audio.volume, 1.0);
        assert!(!audio.connected);
        assert!(!audio.live_enabled);
        assert_eq!(audio.semitone_range, 12);
        assert!(audio.ports.is_empty());
    }

    #[test]
    fn cluster_state_defaults_allow_all_values() {
        use super::ClusterState;
        let cluster = ClusterState::default();
        assert!(cluster.min.is_infinite() && cluster.min < 0.0);
        assert!(cluster.max.is_infinite() && cluster.max > 0.0);
        assert!(!cluster.ego_mode);
        assert!(!cluster.shared_only);
    }

    #[test]
    fn session_state_defaults_are_empty() {
        use super::SessionState;
        let session = SessionState::default();
        assert!(session.name.is_empty());
        assert!(session.saved_sessions.is_empty());
        assert!(session.status.is_none());
    }
}
