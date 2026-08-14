from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from datetime import date, timedelta
from typing import List, Optional
import joblib
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "cycle_model.joblib")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model_metadata.json")

app = FastAPI(title="API Prédiction Cycle Menstruel & Analytics", version="2.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

DUREE_REGLES_DEFAUT = 5
SURVIE_SPERMATOZOIDES = 5
SURVIE_OVULE = 1

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
    # Remplacez avec vos identifiants SMTP réels (Mailtrap, SendGrid, Gmail API, etc.)
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
        "modele": metadata["model_name"],
        "version": "2.5"
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

    X = pd.DataFrame([{
        "cycle_precedent": features["cycle_precedent"],
        "moyenne_3_derniers": features["moyenne_3_derniers"],
        "ecart_type_3_derniers": features["ecart_type_3_derniers"],
        "cycle_number": features["cycle_number"],
    }])

    duree_predite = float(model.predict(X)[0])

    derniere_date_regles = dates[-1]
    date_prochaines_regles = derniere_date_regles + timedelta(days=round(duree_predite))

    phase_luteale = metadata["default_luteal_phase_days"]
    jour_ovulation_estime = date_prochaines_regles - timedelta(days=phase_luteale)
    date_debut_ovulation = jour_ovulation_estime - timedelta(days=2)
    date_fin_ovulation = jour_ovulation_estime + timedelta(days=2)

    phases = construire_phases(
        derniere_date_regles=derniere_date_regles,
        date_prochaines_regles=date_prochaines_regles,
        jour_ovulation_estime=jour_ovulation_estime,
        duree_regles=request.duree_regles,
    )

    # Calculs du jour actuel
    aujourdhui = date.today()
    jour_actuel_cycle = (aujourdhui - derniere_date_regles).days + 1
    jours_avant_prochaines_regles = (date_prochaines_regles - aujourdhui).days

    # Trouver la phase actuelle
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

    # Stats du Dashboard
    stats = StatsDashboard(
        moyenne_duree_cycle=round(float(np.mean(durees_valides)), 1),
        moyenne_duree_regles=float(request.duree_regles),
        ecart_type_cycle=round(float(np.std(durees_valides)), 1),
        nombre_cycles_enregistres=len(durees_valides)
    )

    # Déclencher envoi d'email de rappel si demandé
    if request.envoyer_email_rappel and request.user_email:
        background_tasks.add_task(
            envoyer_email_notification,
            request.user_email,
            date_prochaines_regles
        )

    mae = metadata["mae_days"]
    explication_marge = (
        f"Sur les cycles de test, la prédiction varie en moyenne de {mae} jours. "
        f"Considérez les dates comme le centre d'une fourchette d'environ ±{round(mae)} jours."
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
        marge_erreur_jours=mae,
        explication_marge_erreur=explication_marge,
        avertissement=avertissement,
    )
