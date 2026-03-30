//! Graduated pitch mapping — maps a magnitude value to a MIDI note number.

/// Configuration for graduated note generation.
#[derive(Debug, Clone)]
pub struct GraduatedConfig {
    /// Max semitone offset above/below `base_note` (default 12 = one octave).
    pub semitone_range: i32,
    /// Magnitude value that maps to the lowest note.
    pub magnitude_min: f64,
    /// Magnitude value that maps to the highest note.
    pub magnitude_max: f64,
}

impl Default for GraduatedConfig {
    fn default() -> Self {
        Self {
            semitone_range: 12,
            magnitude_min: 0.0,
            magnitude_max: 1.0,
        }
    }
}

/// Map a `magnitude` value to a MIDI note in `[0, 127]`.
///
/// `magnitude` is clamped to `[magnitude_min, magnitude_max]` then linearly
/// interpolated from `base_note` (at minimum magnitude) up to
/// `base_note + semitone_range` (at maximum magnitude).
pub fn graduated_note(base_note: u8, magnitude: f64, config: &GraduatedConfig) -> u8 {
    let range = (config.magnitude_max - config.magnitude_min).max(1e-9);
    let t = ((magnitude - config.magnitude_min) / range).clamp(0.0, 1.0);
    // t=0 → base_note, t=1 → base_note + semitone_range
    let offset = (t * config.semitone_range as f64).round() as i32;
    let raw = base_note as i32 + offset;
    raw.clamp(0, 127) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graduated_note_range() {
        let cfg = GraduatedConfig {
            semitone_range: 24,
            magnitude_min: -10.0,
            magnitude_max: 10.0,
        };
        for mag in [-10.0, -5.0, 0.0, 5.0, 10.0] {
            let note = graduated_note(60, mag, &cfg);
            assert!(
                (0..=127).contains(&note),
                "note {note} out of [0,127] for mag {mag}"
            );
        }
    }

    #[test]
    fn test_graduated_note_proportional() {
        let cfg = GraduatedConfig {
            semitone_range: 12,
            magnitude_min: 0.0,
            magnitude_max: 1.0,
        };
        let low = graduated_note(60, 0.0, &cfg);
        let high = graduated_note(60, 1.0, &cfg);
        assert!(
            high >= low,
            "higher magnitude should produce higher or equal note"
        );
        // base_note=60, t=0 → 60, t=1 → 72
        assert_eq!(low, 60);
        assert_eq!(high, 72);
    }
}
