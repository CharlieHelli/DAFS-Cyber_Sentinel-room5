# Rôle Analyste : evaluate(model, X_test, y_test) renvoie un dict de métriques (écrit dans outputs/metrics.json).
# anomaly = classe positive (1), normal = 0. Les 8 clés d'origine sont conservées, d'autres sont ajoutées.

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_recall_curve,
    precision_recall_fscore_support, precision_score, recall_score, roc_auc_score,
)

# Réglages : seuil par défaut, rappels visés pour l'analyse de seuil, nombre de rejeux du bootstrap
DEFAULT_THRESHOLD = 0.5
RECALL_TARGETS = (0.99, 0.995, 0.998)
N_BOOTSTRAP = 200


# Convertit en float Python ; NaN/inf -> None (le JSON ne sait pas écrire NaN)
def _clean(value):
    """Transforme un nombre en float Python propre ; renvoie None si c'est NaN (JSON n'aime pas NaN)."""
    value = float(value)
    return value if np.isfinite(value) else None


# Récupère la décision 0/1 et le score d'anomalie (probabilité) du modèle
def get_scores(model, X_test):
    """Renvoie (y_pred, y_proba) : la décision 0/1 et le « score d'anomalie » entre 0 et 1."""
    y_pred = model.predict(X_test)
    last_step = getattr(model, "named_steps", {}).get("model", model)
    if hasattr(last_step, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(last_step, "decision_function"):
        scores = model.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        y_proba = y_pred.astype(float)
    return y_pred, y_proba


# Les 4 cases de la matrice de confusion : TN, FP, FN, TP
def _confusion_counts(y_true, y_pred):
    """Compte les 4 cases : tn (vrai normal), fp (fausse alarme), fn (intrusion ratée), tp (intrusion attrapée)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


# Division sans plantage si le dénominateur vaut 0
def _safe_div(a, b):
    """Division qui renvoie None au lieu de planter quand le dénominateur vaut 0."""
    return None if b == 0 else float(a / b)


# ECE : les probabilités annoncées sont-elles honnêtes ? (0 = parfait)
# On range les scores en 10 tranches et on compare score moyen et fréquence réelle
def _expected_calibration_error(y_true, y_proba, bins=10):
    """ECE : écart moyen entre « ce que le modèle annonce » et « ce qui arrive vraiment », par tranche de score."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(y_proba, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_proba[mask].mean())
    return float(ece)


# Teste tous les seuils : meilleur F1, et seuil le plus haut qui tient un rappel cible
def _threshold_analysis(y_true, y_proba):
    """Cherche le meilleur seuil pour le F1 et le meilleur seuil pour atteindre un rappel cible."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    p, r = prec[:-1], rec[:-1]
    f1s = 2 * p * r / (p + r + 1e-12)
    best = int(np.argmax(f1s))
    out = {"best_f1": {"threshold": _clean(thr[best]), "f1": _clean(f1s[best]),
                       "precision": _clean(p[best]), "recall": _clean(r[best])}}
    for target in RECALL_TARGETS:
        ok = np.where(r >= target)[0]
        out[f"recall_min_{target}"] = (
            {"threshold": _clean(thr[ok[-1]]), "precision": _clean(p[ok[-1]]), "recall": _clean(r[ok[-1]])}
            if len(ok) else None
        )
    return out


# Intervalle de confiance à 95 % : on rejoue le test n fois en tirant avec remise
def _bootstrap_ci(y_true, y_pred, n=N_BOOTSTRAP, seed=42):
    """Intervalle de confiance à 95 % de f1/rappel/précision : on « rejoue » le test n fois avec remise."""
    rng = np.random.default_rng(seed)
    size = len(y_true)
    notes = {"f1": [], "recall": [], "precision": []}
    for _ in range(n):
        idx = rng.integers(0, size, size)
        yt, yp = y_true[idx], y_pred[idx]
        notes["f1"].append(f1_score(yt, yp, zero_division=0))
        notes["recall"].append(recall_score(yt, yp, zero_division=0))
        notes["precision"].append(precision_score(yt, yp, zero_division=0))
    return {k: [_clean(np.percentile(v, 2.5)), _clean(np.percentile(v, 97.5))] for k, v in notes.items()}


# Fonction appelée par run.py
def evaluate(model, X_test, y_test) -> dict:
    """
    Calcule des métriques de classification binaire (anomaly = classe positive).
    Garde les 8 clés d'origine et en ajoute : confusion, taux d'erreur, calibration,
    métriques par classe, analyse de seuil, intervalle de confiance.
    """
    # Vérité et prédictions du modèle ; two_classes évite les erreurs si une classe manque
    y_true = np.asarray(y_test).astype(int)
    y_pred, y_proba = get_scores(model, X_test)
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    two_classes = len(np.unique(y_true)) == 2

    # 1) Métriques d'origine
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if two_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "positives_test": int(np.sum(y_true == 1)),
        "negatives_test": int(np.sum(y_true == 0)),
    }

    # 2) Matrice de confusion et taux d'erreur (FNR = intrusions ratées, le plus critique)
    metrics.update({
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "specificity": _safe_div(tn, tn + fp),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "npv": _safe_div(tn, tn + fn),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    })

    # 3) Qualité des probabilités (calibration)
    metrics.update({
        "log_loss": _clean(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier_score": _clean(brier_score_loss(y_true, y_proba)),
        "expected_calibration_error": _clean(_expected_calibration_error(y_true, y_proba)),
    })

    # 4) Métriques par classe (normal / anomaly)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    metrics["per_class"] = {
        name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, name in enumerate(["normal", "anomaly"])
    }

    # 5) Analyse de seuil et intervalle de confiance
    if two_classes:
        metrics["threshold_analysis"] = _threshold_analysis(y_true, y_proba)
        metrics["default_threshold"] = DEFAULT_THRESHOLD
        metrics["ci95"] = _bootstrap_ci(y_true, np.asarray(y_pred).astype(int))

    return metrics
