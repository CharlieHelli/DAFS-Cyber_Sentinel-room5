# Rôle Data Visualizer : make_plots(...) enregistre les graphiques PNG et deux CSV d'erreurs dans outputs/.
# Couleurs constantes : bleu = normal, orange = anomaly. Un graphique qui échoue affiche un avertissement.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay, auc, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve,
)

# Palette (lisible par les daltoniens), colonnes catégorielles et seuil par défaut
COLOR_NORMAL = "#1c7ed6"
COLOR_ANOMALY = "#e8590c"
COLOR_NEUTRAL = "#868e96"
CATEGORICAL = ["protocol_type", "service", "flag"]
THRESHOLD = 0.5


# Crée le dossier de sortie s'il n'existe pas
def _ensure(outdir: str):
    os.makedirs(outdir, exist_ok=True)


# Enregistre la figure en PNG puis la ferme (libère la mémoire)
def _save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=150)
    plt.close(fig)


# Décision 0/1 et score d'anomalie du modèle
def _scores(model, X_test):
    """Renvoie (y_pred, y_proba) : la décision 0/1 et le score d'anomalie entre 0 et 1."""
    y_pred = np.asarray(model.predict(X_test)).astype(int)
    last_step = getattr(model, "named_steps", {}).get("model", model)
    if hasattr(last_step, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)
    return y_pred, y_proba


# Matrice de confusion : effectifs (gauche) et % par vraie classe (droite)
def plot_confusion(y_true, y_pred, outdir):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    labels = ["normal (0)", "anomaly (1)"]
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title("Effectifs")
    ConfusionMatrixDisplay(cm_pct, display_labels=labels).plot(ax=axes[1], cmap="Blues", colorbar=False, values_format=".1%")
    axes[1].set_title("% par vraie classe")
    tn, fp, fn, tp = cm.ravel()
    fig.suptitle(f"Matrice de confusion — {fn} intrusion(s) ratée(s) (FN), {fp} fausse(s) alarme(s) (FP)", fontsize=11)
    _save(fig, outdir, "confusion_matrix.png")


# Courbe ROC : intrusions détectées vs fausses alarmes ; le point noir = seuil 0.5
def plot_roc(y_true, y_proba, y_pred, outdir):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=COLOR_ANOMALY, lw=2, label=f"modèle (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color=COLOR_NEUTRAL, label="hasard (AUC = 0.5)")
    ax.scatter([fp / (fp + tn)], [tp / (tp + fn)], s=70, color="black", zorder=5, label="seuil 0.5")
    ax.set_xlabel("Taux de fausses alarmes (FPR)")
    ax.set_ylabel("Taux d'intrusions détectées (TPR = rappel)")
    ax.set_title("Courbe ROC")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    _save(fig, outdir, "roc_curve.png")


# Courbe précision-rappel (utile quand les anomalies sont rares) ; aire = average_precision_score
def plot_pr(y_true, y_proba, y_pred, outdir):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    prevalence = float(np.mean(y_true))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color=COLOR_ANOMALY, lw=2, label=f"modèle (AP = {ap:.4f})")
    ax.axhline(prevalence, ls="--", color=COLOR_NEUTRAL, label=f"hasard (précision = {prevalence:.2f})")
    ax.scatter([tp / (tp + fn)], [tp / (tp + fp)], s=70, color="black", zorder=5, label="seuil 0.5")
    ax.set_xlabel("Rappel (intrusions détectées)")
    ax.set_ylabel("Précision (alarmes justifiées)")
    ax.set_title("Courbe précision–rappel")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    _save(fig, outdir, "pr_curve.png")


# Rappel, précision et F1 selon le seuil : aide à choisir le seuil
def plot_threshold_sweep(y_true, y_proba, outdir):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    p, r = prec[:-1], rec[:-1]
    f1 = 2 * p * r / (p + r + 1e-12)
    best = int(np.argmax(f1))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(thr, r, color=COLOR_ANOMALY, lw=2, label="rappel")
    ax.plot(thr, p, color=COLOR_NORMAL, lw=2, label="précision")
    ax.plot(thr, f1, color="black", lw=1.5, ls=":", label="F1")
    ax.axvline(THRESHOLD, color=COLOR_NEUTRAL, ls="--", label="seuil par défaut 0.5")
    ax.scatter([thr[best]], [f1[best]], color="black", zorder=5, label=f"meilleur F1 (seuil {thr[best]:.2f})")
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Valeur")
    ax.set_title("Que se passe-t-il quand on bouge le seuil ?\n(baisser le seuil = plus de rappel, plus de fausses alarmes)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", ncol=2, fontsize=8)
    _save(fig, outdir, "threshold_sweep.png")


# Scores prédits par classe (échelle log pour voir les rares erreurs)
def plot_score_distribution(y_true, y_proba, outdir):
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(y_proba[y_true == 0], bins=bins, alpha=0.7, color=COLOR_NORMAL, label="normal (0)")
    ax.hist(y_proba[y_true == 1], bins=bins, alpha=0.7, color=COLOR_ANOMALY, label="anomaly (1)")
    ax.set_yscale("log")
    ax.axvline(THRESHOLD, color="black", ls="--", label="seuil 0.5")
    ax.set_xlabel("Score d'anomalie prédit")
    ax.set_ylabel("Nombre de connexions (échelle log)")
    ax.set_title("Les deux classes sont-elles bien séparées ?")
    ax.legend()
    _save(fig, outdir, "score_distribution.png")


# Calibration : probabilité annoncée vs fréquence réelle (diagonale = parfait)
def plot_calibration(y_true, y_proba, outdir):
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], "--", color=COLOR_NEUTRAL, label="parfaite")
    ax.plot(mean_pred, frac_pos, "o-", color=COLOR_ANOMALY, lw=2, label="modèle")
    ax.set_xlabel("Probabilité annoncée (moyenne par tranche)")
    ax.set_ylabel("Fréquence réelle d'anomalies")
    ax.set_title("Calibration : peut-on croire les probabilités ?")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    _save(fig, outdir, "calibration_curve.png")


# Variables les plus utiles à XGBoost (vrais noms après encodage, importance « gain »)
def plot_feature_importance(model, outdir, top=15):
    preproc = model.named_steps["preproc"]
    xgb = model.named_steps["model"]
    names = [n.split("__", 1)[-1] for n in preproc.get_feature_names_out()]
    raw = xgb.get_booster().get_score(importance_type="gain")
    scores = {}
    for key, value in raw.items():
        idx = int(key[1:]) if key.startswith("f") and key[1:].isdigit() else None
        scores[names[idx] if idx is not None and idx < len(names) else key] = value
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top]
    if not items:
        return
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.32 * len(items))))
    ax.barh(labels, values, color=COLOR_NORMAL)
    ax.set_xlabel("Gain moyen (importance)")
    ax.set_title(f"Les {len(items)} variables qui comptent le plus (XGBoost)")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, outdir, "xgb_feature_importance.png")


# Ajoute à X_test la vérité, la prédiction et les indicateurs FN / FP
def _error_table(X_test, y_true, y_pred):
    tmp = X_test.copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred
    tmp["is_fn"] = (tmp["y_true"] == 1) & (tmp["y_pred"] == 0)
    tmp["is_fp"] = (tmp["y_true"] == 0) & (tmp["y_pred"] == 1)
    return tmp


# Où le modèle se trompe-t-il ? Erreurs par protocole, flag et service
def plot_error_breakdown(X_test, y_true, y_pred, outdir, top=8):
    tmp = _error_table(X_test, y_true, y_pred)
    cols = [c for c in CATEGORICAL if c in tmp.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 4.4))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        g = tmp.groupby(col).agg(n=("y_true", "size"), fn=("is_fn", "sum"), fp=("is_fp", "sum"))
        g["err"] = g["fn"] + g["fp"]
        g = g.sort_values(["err", "n"], ascending=False).head(top)
        if g["err"].sum() == 0:
            ax.text(0.5, 0.5, "aucune erreur", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(col)
            continue
        g = g[::-1]
        ylabels = [f"{idx} (n={int(row.n)})" for idx, row in g.iterrows()]
        ax.barh(ylabels, g["fn"], color=COLOR_ANOMALY, label="intrusion ratée (FN)")
        ax.barh(ylabels, g["fp"], left=g["fn"], color=COLOR_NORMAL, label="fausse alarme (FP)")
        ax.set_title(f"Erreurs par « {col} »")
        ax.set_xlabel("Nombre d'erreurs")
        ax.grid(axis="x", alpha=0.3)
    handles, lab = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lab, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(outdir, "error_breakdown.png"), dpi=150)
    plt.close(fig)


# Écrit les faux négatifs (intrusions ratées) et faux positifs (fausses alarmes) en CSV avec leur score
def export_errors(X_test, y_true, y_pred, y_proba, outdir):
    tmp = _error_table(X_test, y_true, y_pred)
    tmp["score_anomaly"] = y_proba
    tmp[tmp["is_fn"]].drop(columns=["is_fn", "is_fp"]).sort_values("score_anomaly").to_csv(
        os.path.join(outdir, "false_negatives.csv"), index=True)
    tmp[tmp["is_fp"]].drop(columns=["is_fn", "is_fp"]).sort_values("score_anomaly", ascending=False).to_csv(
        os.path.join(outdir, "false_positives.csv"), index=True)


# Fonction appelée par run.py
def make_plots(model, X_test, y_test, df_full, outdir: str):
    """
    Sauvegarde les PNG : matrice de confusion, ROC, précision-rappel, balayage du seuil,
    distribution des scores, calibration, importance des variables (vrais noms), erreurs par
    catégorie ; et exporte les faux négatifs / faux positifs en CSV.
    Un graphique qui échoue AFFICHE un avertissement au lieu d'être avalé en silence.
    """
    _ensure(outdir)
    y_true = np.asarray(y_test).astype(int)
    y_pred, y_proba = _scores(model, X_test)

    # Chaque graphique est indépendant : si l'un échoue, les autres sont quand même produits
    jobs = [
        ("confusion_matrix.png", lambda: plot_confusion(y_true, y_pred, outdir)),
        ("roc_curve.png", lambda: plot_roc(y_true, y_proba, y_pred, outdir)),
        ("pr_curve.png", lambda: plot_pr(y_true, y_proba, y_pred, outdir)),
        ("threshold_sweep.png", lambda: plot_threshold_sweep(y_true, y_proba, outdir)),
        ("score_distribution.png", lambda: plot_score_distribution(y_true, y_proba, outdir)),
        ("calibration_curve.png", lambda: plot_calibration(y_true, y_proba, outdir)),
        ("xgb_feature_importance.png", lambda: plot_feature_importance(model, outdir)),
        ("error_breakdown.png", lambda: plot_error_breakdown(X_test, y_true, y_pred, outdir)),
        ("false_negatives.csv / false_positives.csv", lambda: export_errors(X_test, y_true, y_pred, y_proba, outdir)),
    ]
    # Exécution des graphiques, avec avertissement en cas d'échec (pas d'erreur silencieuse)
    done = []
    for name, job in jobs:
        try:
            job()
            done.append(name)
        except Exception as exc:
            print(f"[visualize] ATTENTION : {name} non généré -> {type(exc).__name__}: {exc}")
    return done
