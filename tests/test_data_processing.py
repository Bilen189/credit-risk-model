import pandas as pd

from src.data_processing import (
    extract_datetime_features,
    create_aggregate_features,
    create_rfm_features
)


def sample_transactions():
    return pd.DataFrame({
        "TransactionId": ["T1", "T2", "T3"],
        "CustomerId": ["C1", "C1", "C2"],
        "Amount": [100, 200, 50],
        "Value": [100, 200, 50],
        "TransactionStartTime": [
            "2024-01-01 10:00:00",
            "2024-01-02 12:00:00",
            "2024-01-03 15:00:00"
        ]
    })


def test_extract_datetime_features_creates_expected_columns():
    df = sample_transactions()

    result = extract_datetime_features(df)

    expected_columns = {
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year"
    }

    assert expected_columns.issubset(result.columns)


def test_create_aggregate_features_returns_customer_features():
    df = sample_transactions()
    df = extract_datetime_features(df)

    result = create_aggregate_features(df)

    expected_columns = {
        "total_transaction_amount",
        "average_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "total_transaction_value",
        "average_transaction_value"
    }

    assert expected_columns.issubset(result.columns)


def test_create_rfm_features_returns_rfm_columns():
    df = sample_transactions()
    df = extract_datetime_features(df)

    result = create_rfm_features(df)

    expected_columns = {
        "recency_days",
        "frequency",
        "monetary"
    }

    assert expected_columns.issubset(result.columns)