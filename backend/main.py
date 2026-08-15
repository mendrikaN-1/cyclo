"""
API FastAPI pour la prédiction de cycle menstruel, fenêtre d'ovulation et phases
de fertilité.

IMPORTANT — Différence avec les versions précédentes :
Le modèle (régression linéaire) est maintenant codé EN DUR ci-dessous, sous forme
de coefficients numériques, au lieu d'être chargé depuis un fichier `.joblib` externe.

Pourquoi ce choix : sur un environnement serverless comme Vercel, charger un fichier
externe au démarrage (joblib.load) est une source fréquente de plantage silencieux
(FUNCTION_INVOCATION_FAILED) — soit parce que le fichier n'est pas inclus dans le
paquet déployé, soit à cause d'une incompatibilité de version entre scikit-learn
utilisé à l'entraînement et celui installé sur le serveur au moment du déploiement.

Comme notre modèle est une simple régression linéaire (4 coefficients + une
constante), il est largement aussi simple et beaucoup plus robuste de le
recalculer nous-mêmes avec une formule mathématique directe. Zéro dépendance
fichier, zéro risque de désynchronisation de version.

Ces coefficients ont été extraits une fois pour toutes depuis le modèle entraîné
dans model/train_model.py (voir model/model_metadata.json pour la trace complète).
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from datetime import date, timedelta
from typing import List, Optional
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI(title="API Prédiction Cycle Menstruel & Analytics", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# MODÈLE : coefficients de la régression linéaire, extraits une fois
# depuis model/train_model.py. Ordre des features :
# [cycle_precedent, moyenne_3_derniers, ecart_type_3_derniers, cycle_number]
# =====================================================================
MODEL_COEFFICIENTS = np.array([0.30558796, 0.3133756, 0.03538975, -0.02215476])
MODEL_INTERCEPT = 11.212948098682684
MODEL_MAE_DAYS = 2.466  # Erreur moyenne mesurée sur les données de test

DUREE_REGLES_DEFAUT = 5
SURVIE_SPERMATOZOIDES = 5
SURVIE_OVULE = 1
DEFAULT_LUTEAL_PHASE_DAYS = 14


def predire_duree_cycle(cycle_precedent, moyenne_3_derniers, ecart_type_3_derniers, cycle_number):
    """Reproduit exactement model.predict() d'une LinearRegression scikit-learn."""
    features = np.array([cycle_precedent, moyenne_3_derniers, ecart_type_3_derniers, cycle_number])
    return float(np.dot(MODEL_COEFFICIENTS, features) + MODEL_INTERCEPT)


class PredictionRequest(BaseModel):
    dates_dernieres_regles: List[date] = Field(
        ...,
        description="Dates de début de règles passées, de la plus ancienne à la plus récente. Minimum 2 dates."
    )
    duree_regles: int = Field(
        default=DUREE_REGLES_DEFAUT,
        ge=1,
        le=10,
        description="Durée habituelle/réelle des règles en jours."
    )
    user_email: Optional[EmailStr] = None
    envoyer_email_rappel: bool = False


class Phase(BaseModel):
    nom: str
    date_debut: date
    date_fin: date
    niveau_fertilite: str
    conseil: str


class StatsDashboard(BaseModel):
    moyenne_duree_cycle: float
    moyenne_duree_regles: float
    ecart_type_cycle: float
    nombre_cycles_enregistres: int


class PredictionResponse(BaseModel):
    duree_cycle_predite: float
    date_prochaines_regles: date
    date_debut_ovulation: date
    date_fin_ovulation: date
    jour_actuel_cycle: int
    phase_actuelle: Phase
    jours_avant_prochaines_regles: int
    phases: List[Phase]
    stats: StatsDashboard
    nombre_cycles_utilises: int
    marge_erreur_jours: float
    explication_marge_erreur: str
    avertissement: str


def envoyer_email_notification(email_destinataire: str, date_regles_estimee: date):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")

    if not smtp_user or not smtp_password:
        print(f"[SIMULATION EMAIL] Rappel programmé pour {email_destinataire} avant le {date_regles_estimee}")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = email_destinataire
        msg['Subject'] = "Cyclo — Rappel de ton prochain cycle"

        corps = f"""
        Bonjour,

        Selon vos enregistrements sur Cyclo, vos prochaines règles sont estimées pour le {date_regles_estimee.strftime('%d/%m/%Y')}.

        Prenez soin de vous !
        L'équipe Cyclo.
        """
        msg.attach(MIMEText(corps, 'plain'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        print(f"Email de notification envoyé avec succès à {email_destinataire}")
    except Exception as e:
        print(f"Erreur envoi email : {e}")


def calculer_features_depuis_historique(durees_cycles: List[float]) -> dict:
    cycle_precedent = durees_cycles[-1]
    trois_derniers = durees_cycles[-3:]
    moyenne_3_derniers = float(np.mean(trois_derniers))
    ecart_type_3_derniers = float(np.std(trois_derniers)) if len(trois_derniers) > 1 else 0.0
    cycle_number = len(durees_cycles) + 1

    return {
        "cycle_precedent": cycle_precedent,
        "moyenne_3_derniers": moyenne_3_derniers,
        "ecart_type_3_derniers": ecart_type_3_derniers,
        "cycle_number": cycle_number,
    }


def construire_phases(
    derniere_date_regles: date,
    date_prochaines_regles: date,
    jour_ovulation_estime: date,
    duree_regles: int,
) -> List[Phase]:
    debut_fenetre_fertile = jour_ovulation_estime - timedelta(days=SURVIE_SPERMATOZOIDES)
    fin_fenetre_fertile = jour_ovulation_estime + timedelta(days=SURVIE_OVULE)
    fin_menstruelle = derniere_date_regles + timedelta(days=duree_regles - 1)

    phases = [
        Phase(
            nom="Règles",
            date_debut=derniere_date_regles,
            date_fin=fin_menstruelle,
            niveau_fertilite="faible",
            conseil="Grossesse peu probable, mais pas totalement exclue en cas de cycle court.",
        ),
        Phase(
            nom="Phase pré-ovulatoire",
            date_debut=fin_menstruelle + timedelta(days=1),
            date_fin=debut_fenetre_fertile - timedelta(days=1),
            niveau_fertilite="faible",
            conseil="Fertilité en hausse progressive à l'approche de la fenêtre fertile.",
        ),
        Phase(
            nom="Fenêtre fertile (autour de l'ovulation)",
            date_debut=debut_fenetre_fertile,
            date_fin=fin_fenetre_fertile,
            niveau_fertilite="eleve",
            conseil="Période la plus propice à une grossesse.",
        ),
        Phase(
            nom="Phase post-ovulatoire (lutéale)",
            date_debut=fin_fenetre_fertile + timedelta(days=1),
            date_fin=date_prochaines_regles - timedelta(days=1),
            niveau_fertilite="faible",
            conseil="Fertilité en forte baisse jusqu'aux prochaines règles.",
        ),
    ]

    return [p for p in phases if p.date_fin >= p.date_debut]


@app.get("/")
def root():
    return {
        "message": "API de prédiction de cycle menstruel & analytics",
        "modele": "LinearRegression (coefficients intégrés)",
        "version": "3.0"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    dates = sorted(request.dates_dernieres_regles)

    if len(dates) < 2:
        raise HTTPException(
            status_code=400,
            detail="Il faut au moins 2 dates de règles pour calculer un premier cycle.",
        )

    durees_cycles = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
    durees_valides = [d for d in durees_cycles if 15 <= d <= 60]

    if not durees_valides:
        raise HTTPException(
            status_code=400,
            detail="Les dates fournies ne permettent pas de calculer une durée de cycle plausible.",
        )

    features = calculer_features_depuis_historique(durees_valides)

    duree_predite = predire_duree_cycle(
        features["cycle_precedent"],
        features["moyenne_3_derniers"],
        features["ecart_type_3_derniers"],
        features["cycle_number"],
    )

    derniere_date_regles = dates[-1]
    date_prochaines_regles = derniere_date_regles + timedelta(days=round(duree_predite))

    jour_ovulation_estime = date_prochaines_regles - timedelta(days=DEFAULT_LUTEAL_PHASE_DAYS)
    date_debut_ovulation = jour_ovulation_estime - timedelta(days=2)
    date_fin_ovulation = jour_ovulation_estime + timedelta(days=2)

    phases = construire_phases(
        derniere_date_regles=derniere_date_regles,
        date_prochaines_regles=date_prochaines_regles,
        jour_ovulation_estime=jour_ovulation_estime,
        duree_regles=request.duree_regles,
    )

    aujourdhui = date.today()
    jour_actuel_cycle = (aujourdhui - derniere_date_regles).days + 1
    jours_avant_prochaines_regles = (date_prochaines_regles - aujourdhui).days

    phase_actuelle = phases[0]
    for p in phases:
        if p.date_debut <= aujourdhui <= p.date_fin:
            phase_actuelle = p
            break
        elif aujourdhui > phases[-1].date_fin:
            phase_actuelle = Phase(
                nom="Retard potentiel",
                date_debut=date_prochaines_regles,
                date_fin=aujourdhui,
                niveau_fertilite="faible",
                conseil="Prochaines règles en retard par rapport à la prédiction moyenne."
            )

    stats = StatsDashboard(
        moyenne_duree_cycle=round(float(np.mean(durees_valides)), 1),
        moyenne_duree_regles=float(request.duree_regles),
        ecart_type_cycle=round(float(np.std(durees_valides)), 1),
        nombre_cycles_enregistres=len(durees_valides)
    )

    if request.envoyer_email_rappel and request.user_email:
        background_tasks.add_task(
            envoyer_email_notification,
            request.user_email,
            date_prochaines_regles
        )

    explication_marge = (
        f"Sur les cycles de test, la prédiction varie en moyenne de {MODEL_MAE_DAYS} jours. "
        f"Considérez les dates comme le centre d'une fourchette d'environ ±{round(MODEL_MAE_DAYS)} jours."
    )

    avertissement = (
        "Cette estimation est statistique et informative uniquement. Elle ne "
        "constitue pas une méthode contraceptive fiable ni un avis médical."
    )

    return PredictionResponse(
        duree_cycle_predite=round(duree_predite, 1),
        date_prochaines_regles=date_prochaines_regles,
        date_debut_ovulation=date_debut_ovulation,
        date_fin_ovulation=date_fin_ovulation,
        jour_actuel_cycle=jour_actuel_cycle,
        phase_actuelle=phase_actuelle,
        jours_avant_prochaines_regles=jours_avant_prochaines_regles,
        phases=phases,
        stats=stats,
        nombre_cycles_utilises=len(durees_valides),
        marge_erreur_jours=MODEL_MAE_DAYS,
        explication_marge_erreur=explication_marge,
        avertissement=avertissement,
    )
