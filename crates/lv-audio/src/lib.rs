//! `lv-audio` — MIDI engine and beat scheduler for the Lucid Visualization Suite.
//!
//! Enable the `midi` feature flag to get real MIDI output via `midir`.
//! Without the feature, all operations are no-ops.

pub mod graduated;
pub mod midi_engine;
pub mod scheduler;
pub mod soundbank;

pub use graduated::{graduated_note, GraduatedConfig};
pub use midi_engine::{MidiEngine, MidiError};
pub use scheduler::BeatsScheduler;
pub use soundbank::gm_instrument_name;
