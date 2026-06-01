import pandas as pd
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file safely.

    This function checks whether the file exists and whether the loaded
    dataset is empty. This helps prevent silent errors during analysis.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The loaded dataset is empty.")

    return df


def summarize_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a missing value summary table.
    """
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    return pd.DataFrame({
        "missing_values": missing_values,
        "missing_percentage": missing_percentage
    })


def count_duplicates(df: pd.DataFrame) -> int:
    """
    Count duplicate rows in the dataset.
    """
    return int(df.duplicated().sum())