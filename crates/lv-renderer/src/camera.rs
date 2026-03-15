//! Arcball camera.
//!
//! Supports:
//! - Left-drag  → orbit (yaw/pitch)
//! - Right-drag → pan
//! - Scroll     → zoom
//! - Arrow keys → orbit, A/Z keys → zoom, F/S keys → speed
//! - Backspace  → reset, C → centre on scene, Escape → request exit

use nalgebra::{Matrix4, Perspective3, Point3, Vector3};

/// Default field of view (degrees).
const DEFAULT_FOV: f32 = 45.0;
/// Default near clip.
const DEFAULT_NEAR: f32 = 0.1;
/// Default far clip.
const DEFAULT_FAR: f32 = 100_000.0;
/// Maximum pitch (degrees) from horizontal.
const PITCH_LIMIT: f32 = 85.0;
/// Orbit speed (degrees per pixel).
const ORBIT_SPEED: f32 = 0.4;
/// Pan speed (world units per pixel at unit distance).
const PAN_SPEED: f32 = 0.002;
/// Zoom speed (fraction of distance per scroll tick).
const ZOOM_SPEED: f32 = 0.1;
/// Arrow key orbit step (degrees per key event).
const KEY_ORBIT_STEP: f32 = 2.0;
/// A/Z zoom step (fraction of distance per key event).
const KEY_ZOOM_STEP: f32 = 0.05;
/// Default camera speed multiplier.
const DEFAULT_SPEED: f32 = 1.0;

/// Arcball-style camera with yaw/pitch orbit around a target point.
#[derive(Clone)]
pub struct ArcballCamera {
    pub target: Point3<f32>,
    pub distance: f32,
    pub yaw: f32,   // degrees
    pub pitch: f32, // degrees
    pub fov_deg: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
    pub speed: f32,

    // Mouse drag state
    drag_left: Option<(f64, f64)>,
    drag_right: Option<(f64, f64)>,
}

impl ArcballCamera {
    pub fn new(aspect: f32) -> Self {
        Self {
            target: Point3::origin(),
            distance: 800.0,
            yaw: 0.0,
            pitch: 20.0,
            fov_deg: DEFAULT_FOV,
            near: DEFAULT_NEAR,
            far: DEFAULT_FAR,
            aspect,
            speed: DEFAULT_SPEED,
            drag_left: None,
            drag_right: None,
        }
    }

    // ── Matrices ─────────────────────────────────────────────────────────────

    /// Eye position in world space.
    pub fn eye(&self) -> Point3<f32> {
        let yaw_r = self.yaw.to_radians();
        let pitch_r = self.pitch.to_radians();
        let x = self.distance * pitch_r.cos() * yaw_r.sin();
        let y = self.distance * pitch_r.sin();
        let z = self.distance * pitch_r.cos() * yaw_r.cos();
        self.target + Vector3::new(x, y, z)
    }

    /// View matrix (world → camera).
    pub fn view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.eye(), &self.target, &Vector3::y())
    }

    /// Perspective projection matrix.
    pub fn projection_matrix(&self) -> Matrix4<f32> {
        Perspective3::new(self.aspect, self.fov_deg.to_radians(), self.near, self.far)
            .to_homogeneous()
    }

    /// Combined view-projection matrix.
    pub fn view_proj(&self) -> Matrix4<f32> {
        self.projection_matrix() * self.view_matrix()
    }

    // ── Aspect ratio update ───────────────────────────────────────────────────

    pub fn set_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height.max(1) as f32;
    }

    // ── Mouse events ─────────────────────────────────────────────────────────

    pub fn mouse_press_left(&mut self, x: f64, y: f64) {
        self.drag_left = Some((x, y));
    }
    pub fn mouse_release_left(&mut self) {
        self.drag_left = None;
    }
    pub fn mouse_press_right(&mut self, x: f64, y: f64) {
        self.drag_right = Some((x, y));
    }
    pub fn mouse_release_right(&mut self) {
        self.drag_right = None;
    }

    /// Called on every mouse-move event. Returns `true` if camera changed.
    pub fn mouse_moved(&mut self, x: f64, y: f64) -> bool {
        let mut changed = false;

        if let Some((ox, oy)) = self.drag_left.take() {
            let dx = (x - ox) as f32 * ORBIT_SPEED * self.speed;
            let dy = (y - oy) as f32 * ORBIT_SPEED * self.speed;
            self.yaw += dx;
            self.pitch = (self.pitch - dy).clamp(-PITCH_LIMIT, PITCH_LIMIT);
            self.drag_left = Some((x, y));
            changed = true;
        }

        if let Some((ox, oy)) = self.drag_right.take() {
            let dx = (x - ox) as f32 * PAN_SPEED * self.distance * self.speed;
            let dy = (y - oy) as f32 * PAN_SPEED * self.distance * self.speed;
            // Pan in camera's right and up axes
            let yaw_r = self.yaw.to_radians();
            let right = Vector3::new(yaw_r.cos(), 0.0, -yaw_r.sin());
            let up_w = Vector3::y();
            self.target -= right * dx;
            self.target += up_w * dy;
            self.drag_right = Some((x, y));
            changed = true;
        }

        changed
    }

    /// Mouse scroll (positive = zoom in).
    pub fn scroll(&mut self, delta: f32) {
        let factor = 1.0 - delta * ZOOM_SPEED * self.speed;
        self.distance = (self.distance * factor).max(0.1);
    }

    // ── Keyboard events ───────────────────────────────────────────────────────

    /// Handle a key press. Returns `true` if the camera changed, and an
    /// optional `AppAction` signal.
    pub fn key_pressed(&mut self, key: CameraKey) -> (bool, Option<AppAction>) {
        match key {
            CameraKey::Left => {
                self.yaw -= KEY_ORBIT_STEP * self.speed;
                (true, None)
            }
            CameraKey::Right => {
                self.yaw += KEY_ORBIT_STEP * self.speed;
                (true, None)
            }
            CameraKey::Up => {
                self.pitch = (self.pitch + KEY_ORBIT_STEP * self.speed).min(PITCH_LIMIT);
                (true, None)
            }
            CameraKey::Down => {
                self.pitch = (self.pitch - KEY_ORBIT_STEP * self.speed).max(-PITCH_LIMIT);
                (true, None)
            }
            CameraKey::ZoomIn => {
                self.distance *= 1.0 - KEY_ZOOM_STEP * self.speed;
                (true, None)
            }
            CameraKey::ZoomOut => {
                self.distance *= 1.0 + KEY_ZOOM_STEP * self.speed;
                (true, None)
            }
            CameraKey::SpeedUp => {
                self.speed = (self.speed * 1.25).min(10.0);
                (false, None)
            }
            CameraKey::SlowDown => {
                self.speed = (self.speed * 0.8).max(0.05);
                (false, None)
            }
            CameraKey::Reset => {
                self.reset();
                (true, None)
            }
            CameraKey::Centre => {
                self.centre_on_scene();
                (true, None)
            }
            CameraKey::Exit => (false, Some(AppAction::Exit)),
        }
    }

    /// Reset to default camera position.
    pub fn reset(&mut self) {
        self.target = Point3::origin();
        self.distance = 800.0;
        self.yaw = 0.0;
        self.pitch = 20.0;
        self.fov_deg = DEFAULT_FOV;
        self.speed = DEFAULT_SPEED;
    }

    /// Move the target to the centroid of the scene (approximated as origin).
    pub fn centre_on_scene(&mut self) {
        self.target = Point3::origin();
    }
}

/// Camera-relevant key actions (caller maps winit keys to these).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraKey {
    Left,
    Right,
    Up,
    Down,
    ZoomIn,   // A key
    ZoomOut,  // Z key
    SpeedUp,  // F key
    SlowDown, // S key
    Reset,    // Backspace
    Centre,   // C key
    Exit,     // Escape
}

/// Non-camera actions triggered by camera key handler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppAction {
    Exit,
}
