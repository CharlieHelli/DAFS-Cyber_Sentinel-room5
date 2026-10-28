# Tests de l'Analyste (evaluate) et du Data Visualizer (make_plots) : lancer avec « pytest -q »
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from pipeline.evaluate import evaluate
from pipeline.visualize import make_plots

# Anciennes clés de metrics.json : on ne doit JAMAIS les casser
OLD_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "positives_test", "negatives_test"}


@pytest.fixture(scope="module")
def fitted():
    # Mini jeu de données synthétique avec les 3 colonnes texte du vrai projet
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({
        "protocol_type": rng.choice(["tcp", "udp", "icmp"], n),
        "service": rng.choice(["http", "ftp", "private"], n),
        "flag": rng.choice(["SF", "S0", "REJ"], n),
        "src_bytes": rng.integers(0, 1000, n),
        "count": rng.integers(0, 50, n),
    })
    # La cible dépend d'une variable + du bruit : un modèle imparfait (donc des erreurs à analyser)
    y = ((X["flag"] == "S0") | (X["count"] > 40)).astype(int)
    y = pd.Series(np.where(rng.random(n) < 0.05, 1 - y, y))
    pre = ColumnTransformer([("num", StandardScaler(with_mean=False), ["src_bytes", "count"]),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), ["protocol_type", "service", "flag"])])
    model = Pipeline([("preproc", pre), ("model", XGBClassifier(n_estimators=30, max_depth=3, verbosity=0))])
    model.fit(X[:450], y[:450])
    return model, X[450:], y[450:]


def test_evaluate_keeps_old_keys_and_is_json_serializable(fitted):
    model, X, y = fitted
    m = evaluate(model, X, y)
    assert OLD_KEYS <= set(m)
    # json.dumps plante si une valeur n'est pas sérialisable (np.float64, NaN non géré...)
    json.dumps(m)


def test_confusion_counts_add_up(fitted):
    model, X, y = fitted
    m = evaluate(model, X, y)
    c = m["confusion"]
    assert c["tn"] + c["fp"] + c["fn"] + c["tp"] == len(y)
    assert c["tp"] + c["fn"] == m["positives_test"]
    assert c["tn"] + c["fp"] == m["negatives_test"]


def test_rates_are_consistent(fitted):
    model, X, y = fitted
    m = evaluate(model, X, y)
    assert m["false_negative_rate"] == pytest.approx(1 - m["recall"])
    assert m["false_positive_rate"] == pytest.approx(1 - m["specificity"])
    assert 0 <= m["roc_auc"] <= 1 and 0 <= m["pr_auc"] <= 1
    lo, hi = m["ci95"]["f1"]
    assert lo <= hi


def test_make_plots_writes_all_files(fitted, tmp_path):
    model, X, y = fitted
    done = make_plots(model, X, y, None, str(tmp_path))
    # Aucun graphique ne doit avoir échoué en silence
    assert len(done) == 9
    for name in ["confusion_matrix", "roc_curve", "pr_curve", "threshold_sweep", "score_distribution",
                 "calibration_curve", "xgb_feature_importance", "error_breakdown"]:
        assert (tmp_path / f"{name}.png").stat().st_size > 1000
    assert (tmp_path / "false_negatives.csv").exists() and (tmp_path / "false_positives.csv").exists()
