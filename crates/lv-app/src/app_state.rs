//! `app_state` — Ego Cluster system: visibility filtering and edge computation.
//!
//! `EgoClusterState` drives per-frame object visibility and edge generation
//! for the Lucid Visualization Suite's ego-cluster highlighting feature.

use lv_data::{EdgeRow, EtvSheet};
use lv_renderer::GpuEdge;
use std::collections::HashSet;

// ── EgoClusterState ───────────────────────────────────────────────────────────

/// Persistent UI + logic state for the ego-cluster system.
#[derive(Debug, Clone)]
pub struct EgoClusterState {
    /// The currently selected node label (if any).
    pub selected: Option<String>,
    /// Show secondary edges (edges from ego-members to their neighbours).
    pub show_secondary: bool,
    /// Only show objects that appear in ≥2 other nodes' ego clusters.
    pub shared_objects_only: bool,
    /// `cluster_value` lower bound (inclusive).
    pub cluster_value_min: f64,
    /// `cluster_value` upper bound (inclusive).
    pub cluster_value_max: f64,
}

impl Default for EgoClusterState {
    fn default() -> Self {
        Self {
            selected: None,
            show_secondary: false,
            shared_objects_only: false,
            cluster_value_min: f64::NEG_INFINITY,
            cluster_value_max: f64::INFINITY,
        }
    }
}

// ── Visibility ────────────────────────────────────────────────────────────────

/// Compute the set of node labels that should be **fully visible** in the
/// current frame, given the ego-cluster state and cluster-value filter.
///
/// Rules:
/// - No selection: all nodes whose `cluster_value ∈ [min, max]`.
/// - Selection active (no `shared_objects_only`):
///   - Always show the selected node.
///   - Show its direct ego-members (neighbours).
///   - If `show_secondary`: also show members of each neighbour.
/// - `shared_objects_only`: intersect the ego-cluster with nodes that appear
///   in ≥2 distinct other nodes' ego clusters (within the filtered set).
pub fn compute_visible_objects(sheet: &EtvSheet, state: &EgoClusterState) -> HashSet<String> {
    // Build full cluster-filtered label set
    let filtered: HashSet<String> = sheet
        .rows
        .iter()
        .filter(|r| {
            r.cluster_value >= state.cluster_value_min && r.cluster_value <= state.cluster_value_max
        })
        .map(|r| r.label.clone())
        .collect();

    let Some(ref sel) = state.selected else {
        return filtered;
    };

    // Direct ego-members of `sel`
    let direct: HashSet<String> = sheet
        .edges
        .iter()
        .filter_map(|e| {
            if &e.from == sel && filtered.contains(&e.to) {
                Some(e.to.clone())
            } else if &e.to == sel && filtered.contains(&e.from) {
                Some(e.from.clone())
            } else {
                None
            }
        })
        .collect();

    let mut visible: HashSet<String> = HashSet::new();
    visible.insert(sel.clone());
    visible.extend(direct.iter().cloned());

    if state.show_secondary {
        // Add nodes reachable from any direct member
        for member in &direct {
            for e in &sheet.edges {
                if &e.from == member && filtered.contains(&e.to) {
                    visible.insert(e.to.clone());
                } else if &e.to == member && filtered.contains(&e.from) {
                    visible.insert(e.from.clone());
                }
            }
        }
    }

    if state.shared_objects_only {
        // Keep only nodes (excluding selected) that appear in ≥2 other nodes'
        // direct ego-clusters.
        let candidates: Vec<String> = visible
            .iter()
            .filter(|l| l.as_str() != sel.as_str())
            .cloned()
            .collect();

        let mut shared: HashSet<String> = HashSet::new();
        for candidate in &candidates {
            // Count how many distinct other filtered nodes have an edge to candidate
            let count = sheet
                .rows
                .iter()
                .filter(|r| r.label != *candidate && filtered.contains(&r.label))
                .filter(|r| {
                    sheet.edges.iter().any(|e| {
                        (e.from == r.label && e.to == *candidate)
                            || (e.to == r.label && e.from == *candidate)
                    })
                })
                .count();
            if count >= 2 {
                shared.insert(candidate.clone());
            }
        }
        // Always keep the selected node
        shared.insert(sel.clone());
        return shared;
    }

    visible
}

// ── Edge generation ───────────────────────────────────────────────────────────

/// Build the list of `EdgeRow`s to display for an active ego-cluster selection.
///
/// - Primary edges: selected → direct member (white).
/// - Secondary edges (if `show_secondary`): member → their neighbours (grey).
pub fn compute_ego_edges(sheet: &EtvSheet, selected: &str, show_secondary: bool) -> Vec<EdgeRow> {
    let mut result: Vec<EdgeRow> = Vec::new();

    // Collect direct members
    let direct: HashSet<String> = sheet
        .edges
        .iter()
        .filter_map(|e| {
            if e.from == selected {
                Some(e.to.clone())
            } else if e.to == selected {
                Some(e.from.clone())
            } else {
                None
            }
        })
        .collect();

    // Primary edges
    for e in &sheet.edges {
        if e.from == selected || e.to == selected {
            result.push(e.clone());
        }
    }

    if show_secondary {
        for e in &sheet.edges {
            let involves_member = direct.contains(&e.from) || direct.contains(&e.to);
            let involves_selected = e.from == selected || e.to == selected;
            if involves_member && !involves_selected {
                result.push(e.clone());
            }
        }
    }

    result
}

// ── GpuEdge construction ──────────────────────────────────────────────────────

/// Convert edge rows to `GpuEdge`s, resolving positions from the current frame's
/// instance list.  Uses a label→position lookup built from `instances` (paired
/// with `all_labels`).
pub fn build_gpu_edges(
    edge_rows: &[EdgeRow],
    all_labels: &[String],
    positions: &[[f32; 3]],
    selected: &str,
    show_secondary: bool,
) -> Vec<GpuEdge> {
    use std::collections::HashMap;
    let label_to_pos: HashMap<&str, [f32; 3]> = all_labels
        .iter()
        .zip(positions.iter())
        .map(|(l, &p)| (l.as_str(), p))
        .collect();

    edge_rows
        .iter()
        .filter_map(|e| {
            let fp = *label_to_pos.get(e.from.as_str())?;
            let tp = *label_to_pos.get(e.to.as_str())?;

            // Primary: one endpoint is the selected node → white
            let is_primary = e.from == selected || e.to == selected;
            let color = if is_primary || !show_secondary {
                [1.0_f32, 1.0, 1.0, 1.0]
            } else {
                [0.7, 0.7, 0.7, 0.5]
            };
            Some(GpuEdge {
                from_pos: fp,
                to_pos: tp,
                color,
            })
        })
        .collect()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use lv_data::{EdgeRow, EtvRow, EtvSheet};

    fn make_sheet(edges: Vec<(&str, &str)>) -> EtvSheet {
        // Collect all unique labels
        let mut labels: Vec<String> = Vec::new();
        for (a, b) in &edges {
            if !labels.contains(&a.to_string()) {
                labels.push(a.to_string());
            }
            if !labels.contains(&b.to_string()) {
                labels.push(b.to_string());
            }
        }
        let rows: Vec<EtvRow> = labels
            .iter()
            .map(|l| EtvRow {
                label: l.clone(),
                cluster_value: 1.0,
                ..EtvRow::default()
            })
            .collect();
        let edge_rows = edges
            .into_iter()
            .map(|(f, t)| EdgeRow {
                from: f.to_string(),
                to: t.to_string(),
                strength: 1.0,
            })
            .collect();
        EtvSheet {
            name: "test".to_string(),
            sheet_index: 0,
            rows,
            edges: edge_rows,
        }
    }

    #[test]
    fn test_ego_cluster_members() {
        // A has edges to B and C; compute_ego_edges should return those two edges
        let sheet = make_sheet(vec![("A", "B"), ("A", "C"), ("B", "D")]);
        let mut state = EgoClusterState::default();
        state.selected = Some("A".to_string());

        let edges = compute_ego_edges(&sheet, "A", false);
        let targets: HashSet<String> = edges
            .iter()
            .flat_map(|e| [e.from.clone(), e.to.clone()])
            .filter(|l| l != "A")
            .collect();
        assert!(targets.contains("B"), "B should be ego member");
        assert!(targets.contains("C"), "C should be ego member");
        assert!(!targets.contains("D"), "D is not direct member");
    }

    #[test]
    fn test_secondary_edges() {
        // B has edge to D; with show_secondary=true, D should appear in edges
        let sheet = make_sheet(vec![("A", "B"), ("A", "C"), ("B", "D")]);
        let edges = compute_ego_edges(&sheet, "A", true);
        let targets: HashSet<String> = edges
            .iter()
            .flat_map(|e| [e.from.clone(), e.to.clone()])
            .collect();
        assert!(targets.contains("D"), "D should appear via secondary edge");
    }

    #[test]
    fn test_cluster_value_filter() {
        let mut sheet = make_sheet(vec![("A", "B"), ("A", "C")]);
        // Set cluster_value=5 for C so it falls outside [0,4]
        for r in &mut sheet.rows {
            if r.label == "C" {
                r.cluster_value = 5.0;
            }
        }
        let mut state = EgoClusterState::default();
        state.cluster_value_min = 0.0;
        state.cluster_value_max = 4.0;

        let visible = compute_visible_objects(&sheet, &state);
        assert!(
            !visible.contains("C"),
            "C (cluster_value=5) should be filtered out"
        );
        assert!(visible.contains("A") || visible.contains("B"));
    }

    #[test]
    fn test_shared_objects_mode() {
        // A→B, A→C, D→B, D→C — B and C appear in 2 egos; under shared_objects_only
        let sheet = make_sheet(vec![("A", "B"), ("A", "C"), ("D", "B"), ("D", "C")]);
        let mut state = EgoClusterState::default();
        state.selected = Some("A".to_string());
        state.shared_objects_only = true;

        let visible = compute_visible_objects(&sheet, &state);
        // B and C each appear in A's AND D's ego cluster → should be shown
        assert!(visible.contains("B"), "B should be in shared set");
        assert!(visible.contains("C"), "C should be in shared set");
        // A is always kept
        assert!(visible.contains("A"));
    }
}
