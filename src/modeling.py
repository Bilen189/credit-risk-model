import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def prepare_model_data(df: pd.DataFrame, target_column: str):
    """
    Prepare features and target for modeling.
    """
    if target_column not in df.columns:
        raise KeyError(f"{target_column} not found in dataset.")

    drop_columns = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId",
        "CurrencyCode",
        "CountryCode",
        "TransactionStartTime",
        target_column
    ]

    existing_drop_columns = [col for col in drop_columns if col in df.columns]

    X = df.drop(columns=existing_drop_columns)
    y = df[target_column]

    X = X.select_dtypes(include=["number", "bool"])

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.
    """
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate classification model.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc
    }

    return results


def get_confusion_matrix(model, X_test, y_test):
    """
    Return confusion matrix.
    """
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_classification_report(model, X_test, y_test):
    """
    Return classification report.
    """
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)