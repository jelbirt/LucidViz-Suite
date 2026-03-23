use crate::state::{AppState, SessionRequest};

#[derive(Default)]
pub struct SessionPanel;

impl SessionPanel {
    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        ui.horizontal(|ui| {
            ui.label("Name:");
            ui.text_edit_singleline(&mut state.session_name);
        });

        ui.horizontal(|ui| {
            if ui.button("Save session").clicked() {
                let name = state.session_name.trim();
                if !name.is_empty() {
                    state.pending_session_request = Some(SessionRequest::Save(name.to_string()));
                } else {
                    state.session_status = Some("Enter a session name before saving.".into());
                }
            }

            if ui.button("Refresh list").clicked() {
                state.pending_session_request = Some(SessionRequest::RefreshList);
            }
        });

        ui.separator();
        ui.label("Saved sessions");
        egui::ScrollArea::vertical()
            .max_height(120.0)
            .show(ui, |ui| {
                if state.saved_sessions.is_empty() {
                    ui.small("No saved sessions yet.");
                }
                for session in &state.saved_sessions.clone() {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(state.session_name == *session, session)
                            .clicked()
                        {
                            state.session_name = session.clone();
                        }
                        if ui.small_button("Load").clicked() {
                            state.pending_session_request =
                                Some(SessionRequest::Load(session.clone()));
                        }
                    });
                }
            });

        if let Some(status) = &state.session_status {
            ui.separator();
            ui.label(status);
        }
    }
}
