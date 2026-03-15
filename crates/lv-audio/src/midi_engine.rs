//! MIDI engine — wraps `midir` behind the `midi` feature flag.
//!
//! Without the `midi` feature, `MidiEngine::list_ports()` returns `[]` and
//! all note methods are no-ops.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MidiError {
    #[error("MIDI not available: compiled without 'midi' feature")]
    NotAvailable,
    #[error("MIDI port not found: {0}")]
    PortNotFound(String),
    #[cfg(feature = "midi")]
    #[error("midir connect error: {0}")]
    Midir(String),
    #[error("no connection active")]
    NotConnected,
}

// ─────────────────────────────────────────────────────────────────────────────
// With MIDI feature
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "midi")]
mod inner {
    use super::MidiError;
    use midir::{MidiOutput, MidiOutputConnection};

    pub struct MidiEngine {
        connection: Option<MidiOutputConnection>,
        last_program: [u8; 16], // per-channel, to avoid redundant Program Change
    }

    impl MidiEngine {
        pub fn new() -> Self {
            Self {
                connection: None,
                last_program: [255; 16],
            }
        }

        /// Return names of all available MIDI output ports.
        pub fn list_ports() -> Vec<String> {
            match MidiOutput::new("lv-list") {
                Err(_) => vec![],
                Ok(out) => {
                    let ports = out.ports();
                    ports.iter().filter_map(|p| out.port_name(p).ok()).collect()
                }
            }
        }

        /// Connect to the named MIDI output port.
        pub fn connect(&mut self, port_name: &str) -> Result<(), MidiError> {
            let out = MidiOutput::new("lv-engine").map_err(|_| MidiError::NotAvailable)?;
            let ports = out.ports();
            let port = ports
                .iter()
                .find(|p| out.port_name(p).as_deref().unwrap_or("") == port_name)
                .ok_or_else(|| MidiError::PortNotFound(port_name.to_string()))?;
            let conn = out
                .connect(port, "lv-output")
                .map_err(|e| MidiError::Midir(format!("{:?}", e.kind())))?;
            self.connection = Some(conn);
            self.last_program = [255; 16];
            Ok(())
        }

        /// Send note-on for the given channel, note, velocity and instrument.
        pub fn note_on(
            &mut self,
            channel: u8,
            note: u8,
            velocity: u8,
            instrument: u16,
        ) -> Result<(), MidiError> {
            let conn = self.connection.as_mut().ok_or(MidiError::NotConnected)?;
            let ch = channel & 0x0F;
            // Program Change only if instrument is GM (0..127) and changed
            if instrument < 128 {
                let prog = instrument as u8;
                if self.last_program[ch as usize] != prog {
                    let _ = conn.send(&[0xC0 | ch, prog]);
                    self.last_program[ch as usize] = prog;
                }
            } else {
                log::warn!("instrument {instrument} is non-GM; skipping program change");
            }
            conn.send(&[0x90 | ch, note & 0x7F, velocity & 0x7F])
                .map_err(|_| MidiError::NotConnected)
        }

        /// Send note-off.
        pub fn note_off(&mut self, channel: u8, note: u8) -> Result<(), MidiError> {
            let conn = self.connection.as_mut().ok_or(MidiError::NotConnected)?;
            let ch = channel & 0x0F;
            conn.send(&[0x80 | ch, note & 0x7F, 0])
                .map_err(|_| MidiError::NotConnected)
        }

        /// Send All Notes Off on all 16 channels.
        pub fn all_notes_off(&mut self) {
            let Some(conn) = self.connection.as_mut() else {
                return;
            };
            for ch in 0u8..16 {
                let _ = conn.send(&[0xB0 | ch, 0x7B, 0x00]);
            }
        }

        /// Disconnect from the current MIDI port.
        pub fn disconnect(self) {
            if let Some(conn) = self.connection {
                conn.close();
            }
        }
    }

    impl Default for MidiEngine {
        fn default() -> Self {
            Self::new()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Without MIDI feature (stub)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(feature = "midi"))]
mod inner {
    use super::MidiError;

    #[derive(Default)]
    pub struct MidiEngine;

    impl MidiEngine {
        pub fn new() -> Self {
            Self
        }

        pub fn list_ports() -> Vec<String> {
            vec![]
        }

        pub fn connect(&mut self, _port: &str) -> Result<(), MidiError> {
            Err(MidiError::NotAvailable)
        }

        pub fn note_on(
            &mut self,
            _ch: u8,
            _note: u8,
            _vel: u8,
            _inst: u16,
        ) -> Result<(), MidiError> {
            Ok(())
        }

        pub fn note_off(&mut self, _ch: u8, _note: u8) -> Result<(), MidiError> {
            Ok(())
        }

        pub fn all_notes_off(&mut self) {}

        pub fn disconnect(self) {}
    }
}

pub use inner::MidiEngine;
