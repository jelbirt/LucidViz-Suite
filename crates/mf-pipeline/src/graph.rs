//! Build a petgraph co-occurrence graph from NPPMI values.

use petgraph::graph::UnGraph;

use crate::types::{CooccurrenceGraph, MfConfig};

/// Build an undirected petgraph from NPPMI values.
///
/// An edge (i, j) is added when `pmi[i*n+j] > config.min_pmi`.
pub fn build_petgraph(
    graph: &CooccurrenceGraph,
    pmi: &[f64],
    config: &MfConfig,
) -> UnGraph<String, f64> {
    let n = graph.vocab_size;
    let mut pg: UnGraph<String, f64> = UnGraph::new_undirected();

    // Add nodes.
    let nodes: Vec<_> = graph
        .vocab
        .iter()
        .map(|t| pg.add_node(t.0.clone()))
        .collect();

    // Add edges where PMI > threshold.
    for i in 0..n {
        for j in (i + 1)..n {
            let weight = pmi[i * n + j];
            if weight > config.min_pmi {
                pg.add_edge(nodes[i], nodes[j], weight);
            }
        }
    }

    pg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cooccurrence::build_cooccurrence;
    use crate::pmi::compute_pmi;
    use crate::types::{MfConfig, Token};

    #[test]
    fn test_graph_nodes_match_vocab() {
        let tokens: Vec<Token> = ["apple", "banana", "cherry", "apple", "banana"]
            .iter()
            .map(|s| Token(s.to_string()))
            .collect();
        let cfg = MfConfig {
            min_count: 1,
            min_pmi: 0.0,
            window_size: 2,
            ..MfConfig::default()
        };
        let coocc = build_cooccurrence(&tokens, &cfg);
        let pmi = compute_pmi(&coocc);
        let pg = build_petgraph(&coocc, &pmi, &cfg);

        assert_eq!(pg.node_count(), coocc.vocab_size);
    }
}
