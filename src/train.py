import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42
TARGET_COLUMN = "is_high_risk"


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Training dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"{TARGET_COLUMN} not found in dataset.")

    return df


def prepare_features(df: pd.DataFrame):
    drop_columns = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId",
        "CurrencyCode",
        "CountryCode",
        "TransactionStartTime",
        TARGET_COLUMN
    ]

    existing_drop_columns = [col for col in drop_columns if col in df.columns]

    X = df.drop(columns=existing_drop_columns)
    X = X.select_dtypes(include=["number", "bool"])
    y = df[TARGET_COLUMN]

    return X, y


def evaluate_predictions(y_test, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def train_and_log_model(model_name, model, params, X_train, X_test, y_train, y_test):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1
    )

    with mlflow.start_run(run_name=model_name):
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(y_test, y_pred, y_proba)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path=model_name)

        return {
            "model_name": model_name,
            "best_model": best_model,
            "best_params": grid_search.best_params_,
            "metrics": metrics
        }


def main():
    mlflow.set_experiment("credit-risk-model-training")

    data_path = "data/processed/feature_engineered_data.csv"

    df = load_training_data(data_path)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = [
        {
            "name": "logistic_regression",
            "model": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            "params": {
                "C": [0.1, 1.0, 10.0]
            }
        },
        {
            "name": "random_forest",
            "model": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None]
            }
        }
    ]

    results = []

    for item in models:
        result = train_and_log_model(
            item["name"],
            item["model"],
            item["params"],
            X_train,
            X_test,
            y_train,
            y_test
        )
        results.append(result)

    best_result = max(results, key=lambda x: x["metrics"]["roc_auc"])

    print("Best model:", best_result["model_name"])
    print("Best params:", best_result["best_params"])
    print("Metrics:", best_result["metrics"])


if __name__ == "__main__":
    main()