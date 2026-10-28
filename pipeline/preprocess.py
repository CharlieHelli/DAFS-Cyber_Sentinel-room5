import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET = "class"

def preprocess(df: pd.DataFrame) -> Tuple:
    """
    Split into train/test and build a preprocessing pipeline (OHE for cat, scale numerics).
    Returns: X_train, X_test, y_train, y_test, preprocessor
    TODO (Student B): add imputation, feature selection, rare-category bucketing, smarter scaling.
    """
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # detect numeric columns (exclude the known categoricals)
    num_cols = [c for c in X.columns if c not in CATEGORICAL]
    cat_cols = [c for c in CATEGORICAL if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", Pipeline([ 
                ("imp", SimpleImputer(strategy="median")),
                ("sca", StandardScaler(with_mean=False))]), num_cols),       # sparse-friendly
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                 ( "ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,   # keep it sparse if many OHE cols
    )

    # NOTE: we *fit* this in model.train_model to avoid leakage.
    return X_train, X_test, y_train, y_test, preproc
