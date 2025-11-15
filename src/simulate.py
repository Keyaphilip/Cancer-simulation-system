import numpy as np
import networkx as nx

def apply_mutations(G: nx.DiGraph, mutations: dict) -> dict:
    # mutations: {node: activity_override}; e.g., {"BRCA1": -1.0}
    priors = {n: G.nodes[n].get("baseline", 0.0) for n in G.nodes()}
    for n, v in mutations.items():
        if n in G:
            priors[n] = float(v)
    return priors

def propagate(G: nx.DiGraph, priors: dict, steps: int = 5, alpha: float = 0.7) -> dict:
    # Signed message passing: x_{t+1} = alpha * S^T x_t + (1-alpha) * priors
    nodes = list(G.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    S = np.zeros((len(nodes), len(nodes)))
    for u, v, d in G.edges(data=True):
        sign = d.get("sign", +1)
        S[idx[v], idx[u]] += sign
    # row-normalize incoming influences
    row_norms = np.maximum(np.sum(np.abs(S), axis=1, keepdims=True), 1.0)
    S = S / row_norms
    x = np.array([priors[n] for n in nodes], dtype=float)
    for _ in range(steps):
        x = alpha * (S @ x) + (1 - alpha) * np.array([priors[n] for n in nodes])
    return {n: float(x[idx[n]]) for n in nodes}
