import networkx as nx
from typing import Iterable, Tuple

def build_transaction_graph(df: pd.DataFrame, edge_amount_col: str = 'amount_norm') -> nx.DiGraph:
    """
    Construct directed, weighted transaction graph.
    Node id: account id or 'bank:account' composite if needed.
    Edge weight: normalized amount or frequency aggregation.
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        u = f"{row['From Bank']}:{row['From Account']}"
        v = f"{row['To Bank']}:{row['To Account']}"
        w = float(row[edge_amount_col])
        if G.has_edge(u, v):
            G[u][v]['weight'] += w
            G[u][v]['count'] += 1
        else:
            G.add_edge(u, v, weight=w, count=1)
    return G

def extract_subgraph_from_groups(G: nx.DiGraph, groups: Iterable[Tuple], radius: int = 1):
    """
    Given flagged groups (from AntiBenford), relevant subgraphs are extracted.
    'groups' may be account keys or tuples; radius controls neighborhood expansion.
    """
    # implementation: for each node in group, gather neighbors within 'radius' hops and induced subgraph.
    pass
