📖 Contexte
Bienvenue chez Cyber Sentinel, une startup de cybersécurité en pleine croissance qui surveille les réseaux d'entreprise pour détecter d'éventuelles intrusions.

L'équipe Data a travaillé sur un pipeline d'apprentissage automatique interne qui détecte automatiquement le trafic réseau suspect à l'aide d'un ensemble de données de caractéristiques de connexion (ex: protocole, service, octets envoyés, drapeaux…).

Leur prototype fonctionne — mais il est loin d'être prêt pour la production.

Ils font maintenant appel à vous 🫵, un groupe d'analystes de données et d'ingénieurs ML expérimentés, pour améliorer la base de code existante et booster les performances de détection du modèle.

Vous avez reçu l'accès au dépôt :

👉 https://github.com/JedhaBootcamp/DAFS-Cyber_Sentinel/tree/main

L'équipe attend de vous que vous collaboriez comme de véritables ingénieurs : création de branches, revues de code, fusion et maintien d'un historique de projet propre.

🎯 Objectifs de la Mission
Votre objectif est d'améliorer le pipeline actuel de détection d'intrusions de manière collaborative, tout en suivant les pratiques professionnelles Git & GitHub.

Le projet actuellement :

Charge les données de connexion réseau depuis train_data.csv
Exécute une étape de prétraitement (encodage catégoriel, mise à l'échelle, etc.)
Entraîne un modèle XGBoost de référence (baseline)
L'évalue sur un ensemble de test
Exporte les métriques et les graphiques PNG vers /outputs
Votre mission est de rendre ce système meilleur : plus précis, plus robuste, mieux conçu — le tout en travaillant ensemble via Git.

🧩 Aperçu du Dépôt
├── run.py                        # Main entrypoint for the full pipeline
├── pipeline/
│   ├── data.py                   # load_data()         → load CSV, format target
│   ├── preprocess.py             # preprocess()        → split, encode, scale
│   ├── model.py                  # train_model()       → fit XGBoost
│   ├── evaluate.py               # evaluate()          → compute metrics
│   └── visualize.py              # make_plots()        → save PNG charts
├── data/train_data.csv           # Network intrusion dataset
├── outputs/                      # Generated PNGs + metrics.json
└── tests/                        # Basic tests (should all pass)
Copy
L'ensemble de données contient des colonnes telles que :

duration	protocol_type	service	flag	src_bytes	dst_bytes	...	class
0	tcp	http	SF	199	420	...	normal
0	tcp	private	S0	0	0	...	anomaly
La cible class indique si la connexion est normale (0) ou une anomalie (1).

👩‍💻 Votre Rôle dans l'Équipe
Vous travaillerez au sein de la brigade de données collaborative de Cyber Sentinel. Chacun d'entre vous est responsable de l'amélioration d'une fonction clé du pipeline.

Rôle	Fichier	Fonction	Objectif
Data Engineer	data.py	load_data()	Améliorer le chargement, gérer les valeurs NA, assurer la cohérence du schéma
Preprocessing Specialist	preprocess.py	preprocess()	Améliorer l'ingénierie des caractéristiques, l'encodage et la mise à l'échelle
ML Engineer	model.py	train_model()	Optimiser XGBoost, équilibrer les classes, ajouter une logique de validation
Analyst	evaluate.py	evaluate()	Ajouter des métriques, analyser les seuils, améliorer la rigueur de l'évaluation
Data Visualizer	visualize.py	make_plots()	Générer de meilleurs PNG pour la matrice de confusion, ROC, courbes PR
Chaque étudiant sera responsable d'une branche de fonctionnalité (feature branch), mais le succès final dépendra de la fusion et du bon fonctionnement de toutes les branches.

🧭 Flux de travail Git & GitHub
1️⃣ Cloner le dépôt

git clone https://github.com/JedhaBootcamp/DAFS-Cyber_Sentinel.git
cd DAFS-Cyber_Sentinel
Copy
2️⃣ Créer et basculer vers la branche dev

git checkout -b dev
git push -u origin dev
Copy
3️⃣ Créer votre branche de fonctionnalité

git checkout dev
git pull
git checkout -b feature/<your-name>-<function>
Copy
Exemple : feature/alice-preprocess ou feature/bob-model

4️⃣ Implémenter vos améliorations

Exécuter localement :

python run.py
Copy
Vérifier les tests :

pytest -q
Copy
Générer et inspecter outputs/metrics.json et les PNG.

5️⃣ Pousser et ouvrir une Pull Request

Cible de la PR : branche dev
Examiner la PR d'au moins un coéquipier
Résoudre les conflits ensemble (ils sont à prévoir !)
6️⃣ Fusionner toutes les fonctionnalités dans dev

Lorsque tout fonctionne et que les tests passent, fusionner dev → main

Taguer la version :

git tag -a v1.0 -m "First team release"
git push origin v1.0
Copy
🧠 Idées d'amélioration
Gérer avec soin les catégories manquantes ou rares lors du prétraitement
Ajouter de la normalisation ou des pipelines de ColumnTransformer
Utiliser l'arrêt précoce (early stopping) dans XGBoost, ou essayer une recherche de paramètres
Ajuster les seuils pour un rappel plus élevé (critique pour la détection d'intrusion)
Ajouter de nouvelles métriques : précision, rappel, PR-AUC, matrices de confusion
Tracer les principales caractéristiques contribuant à la détection d'anomalies
Sauvegarder plus d'artefacts (ex: faux positifs/négatifs) dans outputs/
✅ Livrables
À la fin de l'exercice, vous devriez avoir :

Un dépôt fonctionnel où :

Tous les tests passent (pytest -q)
Le pipeline s'exécute de bout en bout (python run.py)
De nouvelles métriques et des PNG apparaissent dans /outputs/
Au moins une branche de fonctionnalité par étudiant

Au moins une revue de pull request par étudiant

Une version finale taguée (v1.0) fusionnée dans main

Une courte entrée dans CHANGELOG.md décrivant votre amélioration

⏱️ Calendrier recommandé (3 heures au total)
Phase	Durée	Objectif
1️⃣ Briefing & Setup	20 min	Cloner le repo, comprendre la structure
2️⃣ Implémentation	75 min	Travailler sur votre fonction assignée
3️⃣ Pull Requests & Reviews	45 min	Soumettre et examiner les PR, résoudre les conflits
4️⃣ Intégration & Debrief	40 min	Fusionner dans main, tagger la release, discuter des acquis
🧩 Critères d'Évaluation
Catégorie	Description
Collaboration	Utilisation des branches, commits, PR et revues de code
Qualité du Code	Améliorations propres, documentées et testées
Performance Modèle	Amélioration des métriques de détection (F1, Recall, etc.)
Reproductibilité	Le pipeline s'exécute de bout en bout sans erreur
Travail d'Équipe	Communication et résolution de conflits sur GitHub
