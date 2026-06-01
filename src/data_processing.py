from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The loaded dataset is empty.")

    return df


def summarize_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    return pd.DataFrame({
        "missing_values": missing_values,
        "missing_percentage": missing_percentage
    })


def count_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TransactionStartTime" not in df.columns:
        raise KeyError("TransactionStartTime column not found.")

    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year

    return df


def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["CustomerId", "TransactionId", "Amount", "Value"]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    customer_agg = df.groupby("CustomerId").agg(
        total_transaction_amount=("Amount", "sum"),
        average_transaction_amount=("Amount", "mean"),
        transaction_count=("TransactionId", "count"),
        std_transaction_amount=("Amount", "std"),
        total_transaction_value=("Value", "sum"),
        average_transaction_value=("Value", "mean")
    ).reset_index()

    customer_agg["std_transaction_amount"] = customer_agg[
        "std_transaction_amount"
    ].fillna(0)

    df = df.merge(customer_agg, on="CustomerId", how="left")

    return df


def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TransactionStartTime" not in df.columns:
        raise KeyError("TransactionStartTime column not found.")

    snapshot_date = df["TransactionStartTime"].max()

    rfm = df.groupby("CustomerId").agg(
        last_transaction=("TransactionStartTime", "max"),
        frequency=("TransactionId", "count"),
        monetary=("Amount", "sum")
    ).reset_index()

    rfm["recency_days"] = (
        snapshot_date - rfm["last_transaction"]
    ).dt.days

    df = df.merge(
        rfm[["CustomerId", "recency_days", "frequency", "monetary"]],
        on="CustomerId",
        how="left"
    )

    return df


def build_preprocessing_pipeline(numerical_features, categorical_features):
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessing_pipeline


def process_data(df: pd.DataFrame):
    df = extract_datetime_features(df)
    df = create_aggregate_features(df)
    df = create_rfm_features(df)

    numerical_features = [
        "Amount",
        "Value",
        "PricingStrategy",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
        "total_transaction_amount",
        "average_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "total_transaction_value",
        "average_transaction_value",
        "recency_days",
        "frequency",
        "monetary"
    ]

    categorical_features = [
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId"
    ]

    available_numerical = [
        col for col in numerical_features if col in df.columns
    ]

    available_categorical = [
        col for col in categorical_features if col in df.columns
    ]

    pipeline = build_preprocessing_pipeline(
        available_numerical,
        available_categorical
    )

    processed_array = pipeline.fit_transform(df)

    return processed_array, pipeline, df