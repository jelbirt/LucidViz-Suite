/// Round-trip integration test for lv-data XLSX and JSON I/O.
///
/// This test exercises the full read → validate → write → re-read cycle
/// using synthetic in-memory data (no fixture file required in Phase 1).

#[cfg(test)]
mod tests {
    use lv_data::{
        json_io::{read_etv_json_bytes, write_etv_json_bytes},
        validate_dataset,
        xlsx_reader::read_etv_xlsx_bytes,
        xlsx_writer::write_etv_xlsx_bytes,
        EdgeRow, EtvDataset, EtvRow, EtvSheet, ShapeKind,
    };

    fn two_sheet_dataset() -> EtvDataset {
        let make_row = |label: &str, x: f64, shape: ShapeKind| EtvRow {
            label: label.into(),
            x,
            y: 0.0,
            z: 0.0,
            size: 1.0,
            size_alpha: 0.0,
            spin_x: 0.0,
            spin_y: 0.0,
            spin_z: 0.0,
            shape,
            color_r: 0.5,
            color_g: 0.5,
            color_b: 0.5,
            note: 60,
            instrument: 0,
            channel: 0,
            velocity: 64,
            cluster_value: 0.0,
            beats: 0,
        };

        let sheet0 = EtvSheet {
            name: "T0".into(),
            sheet_index: 0,
            rows: vec![
                make_row("Alpha", 1.0, ShapeKind::Sphere),
                make_row("Beta", -1.0, ShapeKind::Cube),
                make_row("Gamma", 0.0, ShapeKind::Torus),
            ],
            edges: vec![
                EdgeRow {
                    from: "Alpha".into(),
                    to: "Beta".into(),
                    strength: 0.8,
                },
                EdgeRow {
                    from: "Beta".into(),
                    to: "Gamma".into(),
                    strength: 0.4,
                },
            ],
        };

        let sheet1 = EtvSheet {
            name: "T1".into(),
            sheet_index: 1,
            rows: vec![
                make_row("Alpha", 2.0, ShapeKind::Sphere),
                make_row("Beta", -2.0, ShapeKind::Cube),
                make_row("Gamma", 1.0, ShapeKind::Torus),
            ],
            edges: vec![EdgeRow {
                from: "Alpha".into(),
                to: "Gamma".into(),
                strength: 0.6,
            }],
        };

        EtvDataset {
            source_path: None,
            sheets: vec![sheet0, sheet1],
            all_labels: vec!["Alpha".into(), "Beta".into(), "Gamma".into()],
        }
    }

    // ── XLSX round-trip ───────────────────────────────────────────────────────

    #[test]
    fn xlsx_round_trip_two_sheets() {
        let original = two_sheet_dataset();

        // Validate before writing
        validate_dataset(&original).expect("original dataset should be valid");

        // Write to bytes
        let bytes = write_etv_xlsx_bytes(&original).expect("write_etv_xlsx_bytes");
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[..2], b"PK", "XLSX must start with ZIP magic bytes");

        // Read back
        let recovered = read_etv_xlsx_bytes(&bytes).expect("read_etv_xlsx_bytes");

        assert_eq!(recovered.sheets.len(), 2, "should have 2 sheets");

        // Sheet 0
        let s0 = &recovered.sheets[0];
        assert_eq!(s0.name, "T0");
        assert_eq!(s0.rows.len(), 3);
        assert_eq!(s0.rows[0].label, "Alpha");
        assert_eq!(s0.rows[0].shape, ShapeKind::Sphere);
        assert!((s0.rows[0].x - 1.0).abs() < 1e-9);
        assert_eq!(s0.rows[1].shape, ShapeKind::Cube);
        assert_eq!(s0.rows[2].shape, ShapeKind::Torus);

        assert_eq!(s0.edges.len(), 2);
        assert_eq!(s0.edges[0].from, "Alpha");
        assert_eq!(s0.edges[0].to, "Beta");
        assert!((s0.edges[0].strength - 0.8).abs() < 1e-9);

        // Sheet 1
        let s1 = &recovered.sheets[1];
        assert_eq!(s1.name, "T1");
        assert_eq!(s1.rows.len(), 3);
        assert!((s1.rows[0].x - 2.0).abs() < 1e-9);
        assert_eq!(s1.edges.len(), 1);
        assert_eq!(s1.edges[0].from, "Alpha");
        assert_eq!(s1.edges[0].to, "Gamma");

        // Validate recovered dataset
        validate_dataset(&recovered).expect("recovered dataset should be valid");
    }

    #[test]
    fn xlsx_all_shape_kinds_survive_round_trip() {
        let shapes = ShapeKind::ALL;
        let rows: Vec<EtvRow> = shapes
            .iter()
            .enumerate()
            .map(|(i, &shape)| EtvRow {
                label: format!("node_{i}"),
                x: i as f64,
                size: 1.0,
                velocity: 64,
                shape,
                ..EtvRow::default()
            })
            .collect();

        let ds = EtvDataset {
            source_path: None,
            sheets: vec![EtvSheet {
                name: "shapes".into(),
                sheet_index: 0,
                rows,
                edges: vec![],
            }],
            all_labels: shapes
                .iter()
                .enumerate()
                .map(|(i, _)| format!("node_{i}"))
                .collect(),
        };

        let bytes = write_etv_xlsx_bytes(&ds).expect("write");
        let recovered = read_etv_xlsx_bytes(&bytes).expect("read");

        for (i, &shape) in shapes.iter().enumerate() {
            assert_eq!(
                recovered.sheets[0].rows[i].shape, shape,
                "shape round-trip failed for {shape}"
            );
        }
    }

    #[test]
    fn xlsx_midi_fields_survive_round_trip() {
        let row = EtvRow {
            label: "miditest".into(),
            x: 0.0,
            size: 1.0,
            note: 127,
            instrument: 365,
            channel: 15,
            velocity: 1,
            ..EtvRow::default()
        };
        let ds = EtvDataset {
            source_path: None,
            sheets: vec![EtvSheet {
                name: "S".into(),
                sheet_index: 0,
                rows: vec![row],
                edges: vec![],
            }],
            all_labels: vec!["miditest".into()],
        };
        let bytes = write_etv_xlsx_bytes(&ds).expect("write");
        let recovered = read_etv_xlsx_bytes(&bytes).expect("read");
        let r = &recovered.sheets[0].rows[0];
        assert_eq!(r.note, 127);
        assert_eq!(r.instrument, 365);
        assert_eq!(r.channel, 15);
        assert_eq!(r.velocity, 1);
    }

    // ── JSON round-trip ───────────────────────────────────────────────────────

    #[test]
    fn json_round_trip_two_sheets() {
        let original = two_sheet_dataset();
        let bytes = write_etv_json_bytes(&original).expect("write_etv_json_bytes");
        let recovered = read_etv_json_bytes(&bytes).expect("read_etv_json_bytes");

        assert_eq!(recovered.sheets.len(), original.sheets.len());
        for (orig_sheet, rec_sheet) in original.sheets.iter().zip(recovered.sheets.iter()) {
            assert_eq!(orig_sheet.name, rec_sheet.name);
            assert_eq!(orig_sheet.rows.len(), rec_sheet.rows.len());
            assert_eq!(orig_sheet.edges.len(), rec_sheet.edges.len());
            for (o, r) in orig_sheet.rows.iter().zip(rec_sheet.rows.iter()) {
                assert_eq!(o.label, r.label);
                assert_eq!(o.shape, r.shape);
                assert!((o.x - r.x).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn directed_edges_survive_xlsx_and_json_round_trip() {
        let dataset = EtvDataset {
            source_path: None,
            sheets: vec![EtvSheet {
                name: "T0".into(),
                sheet_index: 0,
                rows: vec![
                    EtvRow {
                        label: "Alpha".into(),
                        size: 1.0,
                        velocity: 64,
                        ..EtvRow::default()
                    },
                    EtvRow {
                        label: "Beta".into(),
                        size: 1.0,
                        velocity: 64,
                        ..EtvRow::default()
                    },
                ],
                edges: vec![
                    EdgeRow {
                        from: "Alpha".into(),
                        to: "Beta".into(),
                        strength: 0.8,
                    },
                    EdgeRow {
                        from: "Beta".into(),
                        to: "Alpha".into(),
                        strength: 0.3,
                    },
                ],
            }],
            all_labels: vec!["Alpha".into(), "Beta".into()],
        };

        let xlsx_bytes = write_etv_xlsx_bytes(&dataset).expect("write xlsx");
        let xlsx_recovered = read_etv_xlsx_bytes(&xlsx_bytes).expect("read xlsx");
        assert_eq!(xlsx_recovered.sheets[0].edges, dataset.sheets[0].edges);

        let json_bytes = write_etv_json_bytes(&dataset).expect("write json");
        let json_recovered = read_etv_json_bytes(&json_bytes).expect("read json");
        assert_eq!(json_recovered.sheets[0].edges, dataset.sheets[0].edges);
    }

    // ── LIS buffer estimation ─────────────────────────────────────────────────

    #[test]
    fn lis_buffer_estimate_two_sheets() {
        let ds = two_sheet_dataset();
        // (2-1) * 30 = 30 frames, 3 union labels, 64 bytes each = 5760
        let est = ds.estimated_lis_buffer_bytes(30);
        assert_eq!(est, 30 * 3 * 64);
    }

    #[test]
    fn lis_buffer_estimate_single_sheet_counts_static_frames() {
        let mut ds = two_sheet_dataset();
        ds.sheets.truncate(1);
        ds.all_labels = ds.canonical_all_labels();
        assert_eq!(
            ds.estimated_lis_buffer_bytes(30),
            30 * ds.all_labels.len() * 64
        );
    }
}
