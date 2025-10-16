"""
Top-level orchestration script.
"""

from src.io_utils import load_transactions
from src.preprocessing import normalize_amounts
from src.anti_benford import extract_antibenford_scores
from src.graph_builder import build_transaction_graph, extract_subgraph_from_groups
from src.community.louvain import LouvainDetector
from src.community.parallel_greedy import ParallelGreedy

def main(config):
    # 1. Load
    df = load_transactions(config['data_path'])

    # 2. Preprocess
    df = normalize_amounts(df)

    # 3. AntiBenford extraction
    ab_scores = extract_antibenford_scores(df, groupby_cols=['From Bank','From Account'])

    # 4. Select flagged groups
    flagged = [k for k, chi2, n in ab_scores if chi2 > config['theta_ab']]

    # 5. Graph build
    G = build_transaction_graph(df)

    # 6. Extract subgraphs and run community detection (Louvain or ParallelGreedy)
    subgraphs = extract_subgraph_from_groups(G, flagged, radius=config.get('radius',1))
    results = {}
    for sg in subgraphs:
        # Optionally run ParallelGreedy first as filter, then Louvain
        pg = ParallelGreedy(n_workers=config['n_workers'])
        # community_pg = pg.fit(sg)  # optional: plug-in implementation
        lv = LouvainDetector(resolution=config['resolution'])
        community_lv = lv.fit(sg)
        results[sg] = community_lv

    # 7. Post-process and evaluate
    # Save communities, compute metrics, and produce figures (ROC, PR, confusion matrix).
