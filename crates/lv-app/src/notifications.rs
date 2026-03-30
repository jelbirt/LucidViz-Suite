//! `notifications` — in-app toast notification system.
//!
//! Notifications appear as a vertical stack of toasts in the top-right corner
//! of the viewport.  Each toast auto-dismisses after 5 seconds.

use std::time::{Duration, Instant};

// ── Notification ──────────────────────────────────────────────────────────────

/// Severity level of a notification.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotifLevel {
    Info,
    Warning,
    Error,
}

/// A single in-app toast notification.
#[derive(Debug, Clone)]
pub struct Notification {
    /// Human-readable message.
    pub message: String,
    /// Severity level.
    pub level: NotifLevel,
    /// When this notification expires and should be removed.
    pub expires_at: Instant,
}

impl Notification {
    #[allow(dead_code)]
    const LIFETIME: Duration = Duration::from_secs(5);

    /// Create an info-level notification.
    #[allow(dead_code)]
    pub fn info(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            level: NotifLevel::Info,
            expires_at: Instant::now() + Self::LIFETIME,
        }
    }

    /// Create a warning-level notification.
    #[allow(dead_code)]
    pub fn warn(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            level: NotifLevel::Warning,
            expires_at: Instant::now() + Self::LIFETIME,
        }
    }

    /// Create an error-level notification.
    #[allow(dead_code)]
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            level: NotifLevel::Error,
            expires_at: Instant::now() + Self::LIFETIME,
        }
    }

    /// Returns `true` if this notification has expired.
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }
}

// ── NotificationQueue ─────────────────────────────────────────────────────────

/// Manages a list of active notifications and renders them as egui toasts.
#[derive(Debug, Default)]
pub struct NotificationQueue {
    pub toasts: Vec<Notification>,
}

impl NotificationQueue {
    #[allow(dead_code)]
    pub fn push(&mut self, notif: Notification) {
        self.toasts.push(notif);
    }

    /// Convenience: push an error notification derived from an `anyhow::Error`.
    #[allow(dead_code)]
    pub fn notify_error(&mut self, err: &anyhow::Error) {
        self.push(Notification::error(format!("{err:#}")));
    }

    /// Remove all expired notifications.
    pub fn expire(&mut self) {
        self.toasts.retain(|n| !n.is_expired());
    }

    /// Render toasts into the top-right corner of the egui viewport.
    /// Call once per frame from the egui closure.
    pub fn show(&mut self, ctx: &egui::Context) {
        self.expire();
        if self.toasts.is_empty() {
            return;
        }

        #[allow(deprecated)]
        let screen = ctx.screen_rect();
        let margin = 12.0_f32;

        // Stack from top-right, top-to-bottom
        let mut offset_y = margin;
        for (idx, notif) in self.toasts.iter().enumerate() {
            let (bg, text_col) = match notif.level {
                NotifLevel::Info => (egui::Color32::from_rgb(40, 80, 40), egui::Color32::WHITE),
                NotifLevel::Warning => (egui::Color32::from_rgb(100, 80, 20), egui::Color32::WHITE),
                NotifLevel::Error => (egui::Color32::from_rgb(120, 30, 30), egui::Color32::WHITE),
            };

            let toast_width = 320.0_f32;
            let anchor_x = screen.right() - toast_width - margin;
            let anchor_y = screen.top() + offset_y;

            let id_str = format!(
                "notif_{}_{:?}_{}",
                idx,
                notif.level,
                &notif.message[..notif.message.len().min(20)]
            );
            egui::Window::new(&id_str)
                .id(egui::Id::new(id_str.clone()))
                .fixed_pos(egui::pos2(anchor_x, anchor_y))
                .fixed_size(egui::vec2(toast_width, 0.0))
                .title_bar(false)
                .resizable(false)
                .collapsible(false)
                .frame(
                    egui::Frame::window(&ctx.style())
                        .fill(bg)
                        .inner_margin(egui::Margin::same(8)),
                )
                .show(ctx, |ui| {
                    ui.colored_label(text_col, &notif.message);
                });

            offset_y += 60.0; // approximate toast height + gap
        }
    }
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notification_expires() {
        let n = Notification {
            message: "test".to_string(),
            level: NotifLevel::Info,
            expires_at: Instant::now() - Duration::from_secs(1),
        };
        assert!(n.is_expired());
    }

    #[test]
    fn notification_queue_expire_removes_old() {
        let mut q = NotificationQueue::default();
        q.push(Notification {
            message: "old".to_string(),
            level: NotifLevel::Info,
            expires_at: Instant::now() - Duration::from_secs(1),
        });
        q.push(Notification::info("fresh"));
        q.expire();
        assert_eq!(q.toasts.len(), 1);
        assert_eq!(q.toasts[0].message, "fresh");
    }
}
