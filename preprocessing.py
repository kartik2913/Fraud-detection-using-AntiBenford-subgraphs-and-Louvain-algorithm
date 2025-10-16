import pandas as pd
import numpy as np

def normalize_amounts(df: pd.DataFrame, amount_col: str = 'Amount Received'):
    """
    Normalize amounts (log-scale then min-max) and add a 'first_digit' column.
    """
    df = df.copy()
    df['amount_log'] = np.log1p(df[amount_col].abs())
    df['amount_norm'] = (df['amount_log'] - df['amount_log'].min()) / (df['amount_log'].max() - df['amount_log'].min())
    df['first_digit'] = df[amount_col].astype(str).str.strip().str.replace(r'[^0-9]', '', regex=True).str[0].astype(int)
    return df
