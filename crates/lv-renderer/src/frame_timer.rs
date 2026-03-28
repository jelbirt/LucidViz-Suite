//! Frame advance timer.
//!
//! Tracks wall-clock time and converts it into discrete LIS slice advances,
//! applying an optional target FPS cap and a speed multiplier.

use std::time::Instant;

/// Returned by `FrameTimer::tick()`.
pub struct FrameAdvance {
    /// How many LIS slices to advance this tick (may be 0 if behind).
    pub advance_slices: u32,
    /// Wall-clock delta since last tick (seconds).
    pub dt_secs: f32,
}

/// Controls how fast the LIS animation plays.
pub struct FrameTimer {
    /// Target frames-per-second.  `None` means free-running (vsync-locked).
    pub target_fps: Option<u32>,
    /// Speed multiplier applied on top of `target_fps`.
    pub speed: f32,

    last_tick: Instant,
    accumulated_slices: f32,
}

impl FrameTimer {
    pub fn new() -> Self {
        Self {
            target_fps: None,
            speed: 1.0,
            last_tick: Instant::now(),
            accumulated_slices: 0.0,
        }
    }

    /// Update `target_fps`.
    pub fn set_target_fps(&mut self, fps: Option<u32>) {
        self.target_fps = fps;
    }

    /// Update the speed multiplier.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.max(0.0);
    }

    /// Called once per display refresh. Returns how many slices to advance.
    ///
    /// In vsync mode (no target FPS) always returns 1 slice per call.
    pub fn tick(&mut self) -> FrameAdvance {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32();
        self.last_tick = now;

        let advance_slices = match self.target_fps {
            None => {
                // Free-running: one slice per vsync, scaled by speed.
                // Speed of 0.0 means paused — do not advance.
                if self.speed == 0.0 {
                    0
                } else {
                    self.accumulated_slices += self.speed;
                    let n = self.accumulated_slices.floor() as u32;
                    self.accumulated_slices -= n as f32;
                    n.max(1)
                }
            }
            Some(fps) => {
                // Fixed FPS: accumulate fractional slices.
                let slices_per_second = fps as f32 * self.speed;
                self.accumulated_slices += dt * slices_per_second;
                let n = self.accumulated_slices.floor() as u32;
                self.accumulated_slices -= n as f32;
                n
            }
        };

        FrameAdvance {
            advance_slices,
            dt_secs: dt,
        }
    }
}

impl Default for FrameTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_zero_produces_zero_advance_in_free_running() {
        let mut timer = FrameTimer::new();
        timer.set_speed(0.0);
        // Consume the initial tick to set the baseline.
        let _ = timer.tick();
        let advance = timer.tick();
        assert_eq!(
            advance.advance_slices, 0,
            "speed=0.0 in free-running mode should produce 0 advance slices"
        );
    }

    #[test]
    fn default_speed_produces_at_least_one_advance_in_free_running() {
        let mut timer = FrameTimer::new();
        // default speed is 1.0, target_fps is None (free-running)
        assert_eq!(timer.speed, 1.0);
        assert!(timer.target_fps.is_none());
        let advance = timer.tick();
        assert!(
            advance.advance_slices >= 1,
            "default speed (1.0) in free-running mode should produce advance_slices >= 1, got {}",
            advance.advance_slices
        );
    }
}
