import pandas as pd

EXPECTED_CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET_COL = "class"   # values: "normal" / "anomaly"

def load_data(path: str = "data/train_data.csv") -> pd.DataFrame:
    """
    Load the intrusion dataset and standardize target to {0,1} with column name 'class'.
    1 = anomaly (positive class), 0 = normal.
    TODO (Student A): add dtype hints, NA handling policy, schema validation, and logging.
    """
    df = pd.read_csv(path)

    # basic sanity checks
    missing = [c for c in EXPECTED_CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected categorical columns: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV missing target column '{TARGET_COL}'")

    # Normalize target → binary 0/1
    table_de_correspondance = {
    "normal" : 0,
    "anomaly": 1
    }
    df[TARGET_COL] = df[TARGET_COL].replace(to_replace=table_de_correspondance)
    
    # Conversion de la colonne en booléen
    df[TARGET_COL] = df[TARGET_COL].astype(int).astype(bool)
    # mapping = {"anomaly": 1, "normal": 0}
    # if df[TARGET_COL].dtype == object:
    #     df[TARGET_COL] = df[TARGET_COL].map(mapping)

    # If already numeric, assume 0/1
    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("Target must be 'normal'/'anomaly' or 0/1 after mapping.")

    return df
