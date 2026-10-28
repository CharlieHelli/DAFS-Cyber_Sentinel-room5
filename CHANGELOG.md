# Changelog

## [Non publié]

### Analyst — `pipeline/evaluate.py`
- Ajout de la **matrice de confusion** chiffrée (`confusion`: tn/fp/fn/tp) et des taux d'erreur
  (`specificity`, `false_positive_rate`, `false_negative_rate`, `npv`, `balanced_accuracy`, `mcc`).
- Ajout de la **calibration** (`log_loss`, `brier_score`, `expected_calibration_error`).
- Ajout des métriques **par classe** (`per_class`).
- Ajout de l'**analyse de seuil** (`threshold_analysis`) : meilleur F1 et seuils pour un rappel cible (99 %, 99,5 %, 99,8 %).
- Ajout d'un **intervalle de confiance à 95 %** par bootstrap (`ci95`) pour F1, rappel et précision.
- Les 8 clés d'origine de `metrics.json` sont conservées.

### Data Visualizer — `pipeline/visualize.py`
- **Correction** : `pr_curve.png` n'était jamais généré (`np.trapz` supprimé de NumPy 2, erreur masquée par `except: pass`).
- Nouveaux graphiques : matrice de confusion (effectifs + %), balayage du seuil, distribution des scores,
  calibration, erreurs par `protocol_type` / `service` / `flag`.
- Importance des variables avec les **vrais noms** de colonnes (importance « gain »).
- Export des **faux négatifs / faux positifs** en CSV dans `outputs/`.
- Un graphique qui échoue affiche maintenant un avertissement (plus de `except: pass` silencieux).
- `seaborn` n'est plus requis.

### Tests
- `tests/test_evaluate_visualize.py` : clés conservées, sérialisation JSON, cohérence des taux, génération des fichiers.

### À signaler au Data Engineer (`pipeline/data.py`)
- `dtype == object` ne détecte plus les textes avec pandas 3 : remplacer par `not pd.api.types.is_numeric_dtype(...)`.
