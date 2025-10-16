import numpy as np
from collections import Counter

def benford_expected():
    """Return expected Benford distribution for first digits 1..9."""
    return {d: np.log10(1 + 1/d) for d in range(1,10)}

def chi_square_deviation(obs_counts: dict, total: int) -> float:
    """
    Compute chi-square deviation between observed counts and expected Benford counts.
    obs_counts: dict digit -> count
    total: total observations considered
    """
    expected = benford_expected()
    chi2 = 0.0
    for d in range(1,10):
        O = obs_counts.get(d, 0)
        E = expected[d] * total
        if E > 0:
            chi2 += (O - E)**2 / E
    return chi2

def extract_antibenford_scores(df: pd.DataFrame, groupby_cols: list, min_size: int = 10):
    """
    For each group (e.g., per account or per small subgraph), the chi-square is computed.
    Returns: list of (group_key, chi2, total_count)
    """
    scores = []
    for key, grp in df.groupby(groupby_cols):
        n = len(grp)
        if n < min_size:
            continue
        counts = Counter(grp['first_digit'])
        chi2 = chi_square_deviation(counts, n)
        scores.append((key, chi2, n))
    return scores
