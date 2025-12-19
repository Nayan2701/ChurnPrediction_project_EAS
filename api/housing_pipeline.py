"""
Shared ML pipeline components for the Customer Churn Classification project.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Classifiers (Updated for Churn Prediction)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. Pipeline for Categorical Variables (Geography, Gender)
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

# 2. Pipeline for Numerical Variables
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

def build_preprocessing():
    """Return the ColumnTransformer preprocessing used in the Churn models."""
    return ColumnTransformer(
        [
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
            ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
        ],
        remainder="passthrough",
    )

def make_estimator_for_name(name: str):
    """Return an unconfigured classifier instance."""
    if name == "logistic":
        return LogisticRegression(random_state=42, solver='liblinear')
    elif name == "random_forest":
        return RandomForestClassifier(random_state=42)
    elif name == "histgradientboosting":
        return HistGradientBoostingClassifier(random_state=42)
    elif name == "xgboost":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, use_label_encoder=False, n_jobs=-1)
    elif name == "lightgbm":
        return LGBMClassifier(objective="binary", random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model name: {name}")