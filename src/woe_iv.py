import pandas as pd
import numpy as np


def calculate_woe_iv(df, feature, target):
    data = df[[feature, target]].copy()
    data[feature] = data[feature].astype(str)

    grouped = data.groupby(feature)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    grouped["bad_dist"] = grouped["bad"] / grouped["bad"].sum()
    grouped["good_dist"] = grouped["good"] / grouped["good"].sum()

    grouped["bad_dist"] = grouped["bad_dist"].replace(0, 0.0001)
    grouped["good_dist"] = grouped["good_dist"].replace(0, 0.0001)

    grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
    grouped["iv"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]

    iv_value = grouped["iv"].sum()

    return grouped.reset_index(), iv_value


def calculate_iv_for_features(df, features, target):
    results = []

    for feature in features:
        _, iv_value = calculate_woe_iv(df, feature, target)
        results.append({
            "feature": feature,
            "iv": iv_value
        })

    return pd.DataFrame(results).sort_values(by="iv", ascending=False)