import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def convert_transaction_time(df: pd.DataFrame, date_column: str = "TransactionStartTime") -> pd.DataFrame:
    """
    Convert transaction timestamp column to datetime format.
    """
    df = df.copy()

    if date_column not in df.columns:
        raise KeyError(f"{date_column} not found in dataset.")

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    if df[date_column].isnull().any():
        raise ValueError(f"{date_column} contains invalid datetime values.")

    return df


def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data into customer-level features.
    """
    required_columns = ["CustomerId", "TransactionId", "Amount", "Value", "TransactionStartTime"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    customer_df = df.groupby("CustomerId").agg(
        transaction_count=("TransactionId", "count"),
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        std_amount=("Amount", "std"),
        total_value=("Value", "sum"),
        avg_value=("Value", "mean"),
        first_transaction=("TransactionStartTime", "min"),
        last_transaction=("TransactionStartTime", "max")
    ).reset_index()

    customer_df["std_amount"] = customer_df["std_amount"].fillna(0)

    return customer_df


def create_rfm_features(customer_df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Create Recency, Frequency, and Monetary features.
    """
    customer_df = customer_df.copy()

    if snapshot_date is None:
        snapshot_date = customer_df["last_transaction"].max()

    customer_df["recency_days"] = (snapshot_date - customer_df["last_transaction"]).dt.days
    customer_df["frequency"] = customer_df["transaction_count"]
    customer_df["monetary"] = customer_df["total_amount"]

    return customer_df


def assign_risk_clusters(customer_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Apply K-Means clustering on RFM features.
    """
    customer_df = customer_df.copy()

    rfm_columns = ["recency_days", "frequency", "monetary"]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(customer_df[rfm_columns])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    customer_df["cluster"] = kmeans.fit_predict(rfm_scaled)

    return customer_df


def identify_high_risk_cluster(customer_df: pd.DataFrame) -> int:
    """
    Identify high-risk cluster based on high recency, low frequency, and low monetary value.

    Higher recency means the customer has been inactive longer.
    Lower frequency means fewer transactions.
    Lower monetary value means lower transaction contribution.
    """
    cluster_summary = customer_df.groupby("cluster")[[
        "recency_days",
        "frequency",
        "monetary"
    ]].mean()

    cluster_summary["risk_score"] = (
        cluster_summary["recency_days"].rank(ascending=True)
        + cluster_summary["frequency"].rank(ascending=False)
        + cluster_summary["monetary"].rank(ascending=False)
    )

    high_risk_cluster = cluster_summary["risk_score"].idxmax()

    return int(high_risk_cluster)


def create_proxy_target(customer_df: pd.DataFrame, high_risk_cluster: int) -> pd.DataFrame:
    """
    Create binary proxy target variable.
    """
    customer_df = customer_df.copy()

    customer_df["is_high_risk"] = np.where(
        customer_df["cluster"] == high_risk_cluster,
        1,
        0
    )

    return customer_df


def merge_proxy_target(df: pd.DataFrame, customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge customer-level proxy target back into transaction-level data.
    """
    return df.merge(
        customer_df[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )


def encode_categorical_features(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """
    One-hot encode selected categorical columns.
    """
    df = df.copy()

    available_columns = [col for col in categorical_columns if col in df.columns]

    return pd.get_dummies(
        df,
        columns=available_columns,
        drop_first=True
    )


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed dataset to CSV.
    """
    output_path = str(output_path)
    df.to_csv(output_path, index=False)