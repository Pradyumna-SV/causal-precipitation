"""Tests for consensus → simultaneous panel DAG conversion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from causal_precip.inference import consensus_records_to_panel_dag_edges


def test_drops_self_loops_and_lagged_edges():
    records = [
        {"source": "a", "target": "a", "lag": 1, "pvalue": 0.01},
        {"source": "a", "target": "b", "lag": 1, "pvalue": 0.01},
        {"source": "a", "target": "b", "lag": 0, "pvalue": 0.001},
        {"source": "b", "target": "c", "lag": 0, "pvalue": 0.002},
    ]
    edges = consensus_records_to_panel_dag_edges(records)
    assert ("a", "b") in edges
    assert ("b", "c") in edges
    assert len(edges) == 2


def test_breaks_two_cycle_by_weaker_edge():
    # a→b and b→a at lag 0: only one should survive (stronger p-value retained first, weaker skipped when it closes cycle)
    records = [
        {"source": "a", "target": "b", "lag": 0, "pvalue": 1e-6},
        {"source": "b", "target": "a", "lag": 0, "pvalue": 1e-3},
    ]
    edges = consensus_records_to_panel_dag_edges(records)
    assert len(edges) == 1
    assert edges[0] == ("a", "b")


def test_augment_adds_enso_tp_edge_when_missing():
    import networkx as nx

    from causal_precip.inference import augment_edges_for_treatment_counterfactual

    edges = [("swvl1", "tp")]
    out, aug = augment_edges_for_treatment_counterfactual(edges, "nino34", "tp")
    assert aug
    G = nx.DiGraph()
    G.add_edges_from(out)
    assert nx.has_path(G, "nino34", "tp")


def test_ipw_marginal_propensity_when_no_covariates():
    import pandas as pd

    from causal_precip.inference import _propensity_scores

    df = pd.DataFrame({"T": [0, 1, 0, 1, 1, 0], "Y": [0, 1, 0, 0, 1, 0]})
    ps = _propensity_scores(df, "T", [])
    exp = float(df["T"].mean())
    exp = max(0.01, min(0.99, exp))
    assert abs(ps[0] - exp) < 1e-9
    assert (ps == exp).all()


def test_identification_backbone_precedes_structural_and_keeps_path():
    import networkx as nx

    from causal_precip.inference import identification_dag_edges

    structural = [("u850", "tp")]
    backbone = [
        ("nino34", "z500"),
        ("nino34", "sst"),
        ("z500", "tp"),
        ("sst", "tp"),
        ("tp", "tp_extreme"),
    ]
    edges = identification_dag_edges(structural, backbone)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(G)
    assert nx.has_path(G, "sst", "tp_extreme")
    assert ("tp", "tp_extreme") in edges
