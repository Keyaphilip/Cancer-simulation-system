import pandas as pd
import networkx as nx

EDGE_COLS = ["src", "dst", "sign"]  # sign in {+1, -1}

def load_pathway_csv(path: str) -> nx.DiGraph:
    df = pd.read_csv(path)
    assert set(EDGE_COLS).issubset(df.columns), f"Required cols: {EDGE_COLS}"
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), sign=int(r["sign"]))
    # ensure nodes have default attrs
    for n in G.nodes():
        G.nodes[n].setdefault("baseline", 0.0)
    return G

def make_example_ddr() -> nx.DiGraph:
    # Minimal DDR sketch
    edges = [
        ("DNA_damage", "ATM", +1),
        ("ATM", "BRCA1", +1),
        ("ATM", "p53", +1),
        ("BRCA1", "HR_repair", +1),
        ("BRCA1", "Genomic_instability", -1),
        ("p53", "Apoptosis", +1),
        ("p53", "Cell_cycle_arrest", +1),
        ("HR_repair", "Genomic_instability", -1),
    ]
    G = nx.DiGraph()
    for s, d, sign in edges:
        G.add_edge(s, d, sign=sign)
    for n in G.nodes():
        G.nodes[n]["baseline"] = 0.0
    return G
