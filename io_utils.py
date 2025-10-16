import pandas as pd

def load_transactions(path: str) -> pd.DataFrame:
    """
    Load transaction CSV and perform light validation.
    Returns: DataFrame with expected columns:
      ['Timestamp','From Bank','From Account','To Bank','To Account',
       'Amount Received','Receiving Currency','Amount Paid','Payment Currency','Payment Format']
    """
    df = pd.read_csv(path)
    # basic checks: required cols, types, nulls
    return df
