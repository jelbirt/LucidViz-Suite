use crate::state::{AppState, SessionRequest};

#[derive(Default)]
pub struct SessionPanel;

impl SessionPanel {
    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        ui.horizontal(|ui| {
            ui.label("Name:");
            ui.text_edit_singleline(&mut state.session.name);
        });

        ui.horizontal(|ui| {
            if ui.button("Save session").clicked() {
                let name = state.session.name.trim();
                if !name.is_empty() {
                    state.session.pending_request = Some(SessionRequest::Save(name.to_string()));
                } else {
                    state.session.status = Some("Enter a session name before saving.".into());
                }
            }

            if ui.button("Refresh list").clicked() {
                state.session.pending_request = Some(SessionRequest::RefreshList);
            }
        });

        ui.separator();
        ui.label("Saved sessions");
        egui::ScrollArea::vertical()
            .max_height(120.0)
            .show(ui, |ui| {
                if state.session.saved_sessions.is_empty() {
                    ui.small("No saved sessions yet.");
                }
                for session in &state.session.saved_sessions.clone() {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(state.session.name == *session, session)
                            .clicked()
                        {
                            state.session.name = session.clone();
                        }
                        if ui.small_button("Load").clicked() {
                            state.session.pending_request =
                                Some(SessionRequest::Load(session.clone()));
                        }
                    });
                }
            });

        if let Some(status) = &state.session.status {
            ui.separator();
            ui.label(status);
        }
    }
}
