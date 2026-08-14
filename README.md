# Cyclo — Prédiction de cycle menstruel (V1)

Projet ML + API + frontend pour prédire la durée du prochain cycle menstruel
et la fenêtre d'ovulation, à partir de l'historique de dates de règles.

## Ce qui est inclus

```
projet_cycle/
├── data/
│   └── rawFedCycleData.csv       # Dataset source (Marquette University, FedCycleData)
├── model/
│   ├── train_model.py            # Script d'entraînement (feature engineering + comparaison de modèles)
│   ├── cycle_model.joblib        # Modèle entraîné et sauvegardé
│   └── model_metadata.json       # Métriques et infos du modèle (MAE, features utilisées...)
├── backend/
│   └── main.py                   # API FastAPI (route /predict, phases du cycle)
├── frontend/
│   └── index.html                # Interface (HTML/CSS/JS + client Supabase via CDN)
├── supabase_schema.sql           # Script SQL à exécuter dans Supabase (table + sécurité)
├── requirements.txt
└── README.md
```

## Configurer Supabase (auth + sauvegarde de l'historique)

1. Crée un compte gratuit sur [supabase.com](https://supabase.com), puis un nouveau projet.
2. Dans **SQL Editor**, colle et exécute le contenu de `supabase_schema.sql` — ça crée
   la table `cycles` et les règles de sécurité (chaque utilisatrice ne voit que ses
   propres données, jamais celles des autres).
3. Dans **Project Settings > API**, récupère ton `Project URL` et ta clé `anon public`.
4. Ouvre `frontend/index.html`, tout en haut du `<script>`, remplace :
   ```js
   const SUPABASE_URL = "https://TON-PROJET.supabase.co";
   const SUPABASE_ANON_KEY = "TA-CLE-ANON-PUBLIQUE";
   ```
   par tes vraies valeurs.
5. Par défaut, Supabase demande une **confirmation par email** à l'inscription. Pour
   tester plus vite en développement, tu peux désactiver ça dans **Authentication >
   Providers > Email > Confirm email** (à réactiver avant une mise en production réelle).

Une fois configuré : à l'inscription/connexion, l'utilisatrice peut saisir ses dates,
elles sont automatiquement sauvegardées, et préremplies à sa prochaine visite.

## Comment ça marche (résumé technique)

1. **Le modèle** a été entraîné sur 1665 cycles réels de 159 femmes (dataset FedCycleData).
   Pour chaque cycle, on utilise comme features : la durée du cycle précédent, la moyenne
   et l'écart-type des 3 derniers cycles, et le numéro du cycle. Cible : la durée du cycle
   suivant. Régression linéaire retenue (MAE ≈ 2.47 jours sur les données de test).

2. **Le backend** (FastAPI) reçoit une liste de dates de règles, calcule les durées de
   cycles correspondantes, construit les mêmes features que celles utilisées à
   l'entraînement, et appelle le modèle pour prédire la durée du prochain cycle.
   La fenêtre d'ovulation est déduite à rebours (phase lutéale ≈ 14 jours avant les
   prochaines règles prédites).

3. **Le frontend** est une simple page HTML/JS qui appelle l'API et affiche le résultat.
   Pas de framework pour rester simple à faire tourner et déployer en V1.

## Lancer le projet en local

### 1. Installer les dépendances

```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optionnel) Ré-entraîner le modèle

Le modèle est déjà entraîné et sauvegardé dans `model/cycle_model.joblib`.
Si tu veux le regénérer :

```bash
python model/train_model.py
```

### 3. Lancer le backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

L'API est accessible sur `http://localhost:8000`.
Documentation interactive auto-générée : `http://localhost:8000/docs`

### 4. Lancer le frontend

Le plus simple : ouvre directement `frontend/index.html` dans ton navigateur
(double-clic dessus). Le fichier appelle `http://localhost:8000` par défaut.

Si le navigateur bloque les appels locaux, sers-le avec un petit serveur :
```bash
cd frontend
python -m http.server 5500
```
Puis va sur `http://localhost:5500`.

## Déploiement gratuit (quand tu seras prêt)

| Composant | Service gratuit | Notes |
|---|---|---|
| Backend (FastAPI) | [Render](https://render.com) ou [Railway](https://railway.app) | Free tier suffisant pour un usage perso/démo |
| Frontend (HTML) | [Vercel](https://vercel.com) ou [Netlify](https://netlify.com) | Dépôt statique, déploiement en 1 clic |

Une fois le backend déployé, il faut changer une seule ligne dans
`frontend/index.html` :
```js
const API_URL = "http://localhost:8000";
```
en la remplaçant par l'URL de ton backend déployé (ex: `https://ton-api.onrender.com`).

## Limites connues de cette V2 (assumées)

- **Modèle global uniquement** : la prédiction utilise l'historique de la femme comme
  *input* du modèle, mais le modèle lui-même reste entraîné sur l'ensemble des 159
  femmes du dataset — ce n'est pas encore un modèle réentraîné par utilisatrice.
- **Précision** : MAE ≈ 2.5 jours. Correct pour une V1/V2, mais peut être amélioré avec
  plus de features (ex. symptômes, durée des règles réelle) ou plus de données.
- **Fenêtre fertile approximative** : basée sur des règles biologiques générales
  (survie des spermatozoïdes ~5 jours, de l'ovule ~24h), pas sur des données
  individuelles (température basale, glaire cervicale...). À ne jamais utiliser
  comme méthode contraceptive fiable — l'app le rappelle explicitement.
- **Confirmation email Supabase** : activée par défaut, ce qui veut dire qu'un compte
  test doit avoir accès à une vraie boîte mail pour se connecter la première fois.

## Prochaines étapes possibles

1. Modèle personnalisé par utilisatrice une fois qu'elle a accumulé assez de cycles.
2. Migrer le frontend vers Next.js si tu veux un projet plus proche de ton stack
   habituel (E-DEVY).
3. Explorer des features supplémentaires dans `model/train_model.py` (durée des
   règles, symptômes disponibles dans le dataset FedCycleData).
4. Déploiement (voir section précédente) + variables d'environnement pour les clés
   Supabase au lieu de les coder en dur dans `index.html`.
