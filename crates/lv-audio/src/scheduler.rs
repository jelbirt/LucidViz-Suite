//! Beat scheduler — fires MIDI notes on each LIS frame advance.

use lv_data::{LisFrame, LvRow};
use std::time::Duration;

use crate::graduated::{graduated_note, GraduatedConfig};
use crate::midi_engine::MidiEngine;

/// How data dimensions map to MIDI parameters during beat events.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BeatMapping {
    /// Centrality (size) → graduated pitch; velocity is fixed.
    #[default]
    CentralityToPitch,
    /// Degree (size) → MIDI velocity; pitch stays at base note.
    DegreeToVelocity,
    /// Betweenness (size) → graduated pitch; closeness (cluster_value) → velocity.
    BetweennessPitchClosenessVelocity,
    /// Cluster value → MIDI channel (bucketed 0–15).
    ClusterToChannel,
}

/// A note-off that is pending until a future slice.
struct PendingNoteOff {
    channel: u8,
    note: u8,
    fire_at_frame: u32,
}

/// Drives MIDI note events from LIS frame advances.
pub struct BeatsScheduler {
    engine: MidiEngine,
    pending_offs: Vec<PendingNoteOff>,
    /// How many beats (note events) to fire per LIS transition.
    pub beats: u32,
    /// LIS value (total slices) — needed to compute beat interval.
    pub lis_value: u32,
    /// Velocity for all notes (0–127).
    pub velocity: u8,
    /// Note hold duration in slices.
    pub hold_slices: u32,
}

impl BeatsScheduler {
    /// Create a scheduler with the given [`MidiEngine`].
    pub fn new(engine: MidiEngine) -> Self {
        Self {
            engine,
            pending_offs: vec![],
            beats: 1,
            lis_value: 30,
            velocity: 64,
            hold_slices: 2,
        }
    }

    pub fn list_ports() -> Vec<String> {
        MidiEngine::list_ports()
    }

    pub fn connect(&mut self, port_name: &str) -> Result<(), crate::midi_engine::MidiError> {
        self.engine.connect(port_name)
    }

    pub fn disconnect(&mut self) {
        self.stop();
        let old_engine = std::mem::replace(&mut self.engine, MidiEngine::new());
        old_engine.disconnect();
    }

    pub fn test_tone(&mut self) -> Result<(), crate::midi_engine::MidiError> {
        self.engine.note_on(0, 60, 100, 0)?;
        std::thread::sleep(Duration::from_millis(120));
        self.engine.note_off(0, 60)
    }

    /// Effective beats: `min(beats, lis_value / 2)` to avoid over-firing.
    fn effective_beats(&self) -> u32 {
        let max = (self.lis_value / 2).max(1);
        self.beats.min(max)
    }

    /// Interval between beats in slices.
    fn beat_interval(&self) -> u32 {
        let eb = self.effective_beats();
        if eb == 0 {
            return u32::MAX;
        }
        (self.lis_value / eb).max(1)
    }

    /// Called once per rendered frame. Fires note-ons and note-offs as needed.
    ///
    /// * `frame`      — current LIS frame (provides `local_slice`)
    /// * `rows`       — `LvRow` data for the current time-slice
    /// * `graduated`  — whether to use graduated pitch mapping
    /// * `grad_config`— graduated mapping configuration
    pub fn on_frame_advance(
        &mut self,
        frame: &LisFrame,
        rows: &[LvRow],
        graduated: bool,
        grad_config: &GraduatedConfig,
        mapping: BeatMapping,
    ) {
        let absolute_slice = frame.slice_index;
        let local_slice = frame.local_slice;

        // ── fire pending note-offs ────────────────────────────────────────
        let mut i = 0;
        while i < self.pending_offs.len() {
            if self.pending_offs[i].fire_at_frame <= absolute_slice {
                let off = self.pending_offs.remove(i);
                let _ = self.engine.note_off(off.channel, off.note);
            } else {
                i += 1;
            }
        }

        // ── decide whether to fire a beat on this slice ───────────────────
        let interval = self.beat_interval();
        if interval == u32::MAX || !local_slice.is_multiple_of(interval) {
            return;
        }

        // ── fire note-ons for every row ───────────────────────────────────
        for row in rows {
            let base = row.note;
            let max_vel = self.velocity;

            let (channel, note, vel) = match mapping {
                BeatMapping::CentralityToPitch => {
                    let note = if graduated {
                        graduated_note(base, row.size, grad_config)
                    } else {
                        base
                    };
                    let vel = (row.velocity as u32).min(max_vel as u32) as u8;
                    (row.channel, note, vel)
                }
                BeatMapping::DegreeToVelocity => {
                    let vel = (row.size.clamp(0.0, 1.0) * 127.0).round() as u8;
                    (row.channel, base, vel)
                }
                BeatMapping::BetweennessPitchClosenessVelocity => {
                    let note = if graduated {
                        graduated_note(base, row.size, grad_config)
                    } else {
                        base
                    };
                    let vel = (row.cluster_value.clamp(0.0, 1.0) * 127.0).round() as u8;
                    (row.channel, note, vel)
                }
                BeatMapping::ClusterToChannel => {
                    let ch = (row.cluster_value.abs() as usize % 16) as u8;
                    let note = if graduated {
                        graduated_note(base, row.size, grad_config)
                    } else {
                        base
                    };
                    let vel = (row.velocity as u32).min(max_vel as u32) as u8;
                    (ch, note, vel)
                }
            };

            let _ = self.engine.note_on(channel, note, vel, row.instrument);

            let off_at = absolute_slice + self.hold_slices.max(1);
            self.pending_offs.push(PendingNoteOff {
                channel,
                note,
                fire_at_frame: off_at,
            });
        }
    }

    /// Send All Notes Off and clear pending note-offs.
    pub fn stop(&mut self) {
        self.engine.all_notes_off();
        self.pending_offs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi_engine::MidiEngine;

    fn dummy_frame(local_slice: u32) -> LisFrame {
        LisFrame {
            instances: vec![],
            labels: vec![],
            slice_index: local_slice,
            transition_index: 0,
            local_slice,
        }
    }

    fn dummy_row() -> LvRow {
        LvRow {
            note: 60,
            channel: 0,
            velocity: 64,
            instrument: 0,
            size: 0.5,
            ..LvRow::default()
        }
    }

    fn grad_cfg() -> GraduatedConfig {
        GraduatedConfig::default()
    }

    #[test]
    fn test_beats_fire_at_slice_0() {
        // beats=1, lis=10 → interval=10 → fires at slice 0, 10, 20 ...
        let mut sched = BeatsScheduler::new(MidiEngine::new());
        sched.beats = 1;
        sched.lis_value = 10;

        // Should fire at slice 0
        let rows = vec![dummy_row()];
        sched.on_frame_advance(
            &dummy_frame(0),
            &rows,
            false,
            &grad_cfg(),
            BeatMapping::default(),
        );
        // pending_offs should have 1 entry after firing
        assert_eq!(sched.pending_offs.len(), 1);

        // Should NOT fire at slice 1
        let pending_before = sched.pending_offs.len();
        sched.on_frame_advance(
            &dummy_frame(1),
            &rows,
            false,
            &grad_cfg(),
            BeatMapping::default(),
        );
        // pending_offs unchanged (no new fires, but still holding old one)
        assert_eq!(sched.pending_offs.len(), pending_before);
    }

    #[test]
    fn test_beats_fire_two_per_transition() {
        // beats=2, lis=10 → interval=5 → fires at slices 0 and 5
        let mut sched = BeatsScheduler::new(MidiEngine::new());
        sched.beats = 2;
        sched.lis_value = 10;
        sched.hold_slices = 1;

        let rows = vec![dummy_row()];
        let grad = grad_cfg();

        let mut fires = 0u32;
        for slice in 0u32..10 {
            let before = sched.pending_offs.len();
            sched.on_frame_advance(
                &dummy_frame(slice),
                &rows,
                false,
                &grad,
                BeatMapping::default(),
            );
            let after = sched.pending_offs.len();
            if after > before {
                fires += 1;
            }
        }
        assert_eq!(fires, 2, "expected 2 fires for beats=2, lis=10");
    }

    #[test]
    fn test_beats_constraint_enforced() {
        // beats=6, lis=10 → effective_beats=min(6, 10/2=5)=5 → interval=2
        let sched = BeatsScheduler {
            engine: MidiEngine::new(),
            pending_offs: vec![],
            beats: 6,
            lis_value: 10,
            velocity: 64,
            hold_slices: 1,
        };
        assert_eq!(sched.effective_beats(), 5);
        assert_eq!(sched.beat_interval(), 2);
    }

    #[test]
    fn test_note_offs_follow_absolute_frame_index_across_transitions() {
        let mut sched = BeatsScheduler::new(MidiEngine::new());
        sched.beats = 1;
        sched.lis_value = 10;
        sched.hold_slices = 2;

        let rows = vec![dummy_row()];
        let grad = grad_cfg();

        let firing_frame = LisFrame {
            instances: vec![],
            labels: vec![],
            slice_index: 9,
            transition_index: 0,
            local_slice: 0,
        };
        sched.on_frame_advance(&firing_frame, &rows, false, &grad, BeatMapping::default());
        assert_eq!(sched.pending_offs.len(), 1);

        let next_transition = LisFrame {
            instances: vec![],
            labels: vec![],
            slice_index: 10,
            transition_index: 1,
            local_slice: 0,
        };
        sched.on_frame_advance(
            &next_transition,
            &rows,
            false,
            &grad,
            BeatMapping::default(),
        );
        assert_eq!(sched.pending_offs.len(), 2);

        let release_frame = LisFrame {
            instances: vec![],
            labels: vec![],
            slice_index: 11,
            transition_index: 1,
            local_slice: 1,
        };
        sched.on_frame_advance(&release_frame, &rows, false, &grad, BeatMapping::default());
        assert_eq!(sched.pending_offs.len(), 1);
    }
}
