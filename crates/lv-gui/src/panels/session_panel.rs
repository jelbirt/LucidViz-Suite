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

        ui.horizontal(|ui| {
            ui.label("Saved sessions");
            if state.session.loading {
                ui.spinner();
            }
        });

        egui::ScrollArea::vertical()
            .max_height(150.0)
            .show(ui, |ui| {
                if state.session.saved_sessions.is_empty() {
                    ui.small("No saved sessions yet.");
                }
                for session in &state.session.saved_sessions.clone() {
                    // Check if this session is being renamed.
                    if state.session.renaming.as_deref() == Some(session) {
                        ui.horizontal(|ui| {
                            let response =
                                ui.text_edit_singleline(&mut state.session.rename_buffer);
                            if response.lost_focus()
                                || ui.input(|i| i.key_pressed(egui::Key::Enter))
                            {
                                let new_name = state.session.rename_buffer.trim().to_string();
                                if !new_name.is_empty() && new_name != *session {
                                    state.session.pending_request = Some(SessionRequest::Rename {
                                        from: session.clone(),
                                        to: new_name,
                                    });
                                }
                                state.session.renaming = None;
                            }
                            if ui.small_button("Cancel").clicked() {
                                state.session.renaming = None;
                            }
                        });
                        continue;
                    }

                    // Check if this session is pending delete confirmation.
                    if state.session.confirm_delete.as_deref() == Some(session) {
                        ui.horizontal(|ui| {
                            ui.small("Delete?");
                            if ui.small_button("Yes").clicked() {
                                state.session.pending_request =
                                    Some(SessionRequest::Delete(session.clone()));
                                state.session.confirm_delete = None;
                            }
                            if ui.small_button("No").clicked() {
                                state.session.confirm_delete = None;
                            }
                        });
                        continue;
                    }

                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(state.session.name == *session, session)
                            .clicked()
                        {
                            state.session.name = session.clone();
                        }
                        if ui.small_button("Load").clicked() {
                            state.session.loading = true;
                            state.session.pending_request =
                                Some(SessionRequest::Load(session.clone()));
                        }
                        if ui.small_button("Rename").clicked() {
                            state.session.renaming = Some(session.clone());
                            state.session.rename_buffer = session.clone();
                        }
                        if ui.small_button("Delete").clicked() {
                            state.session.confirm_delete = Some(session.clone());
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
