"""
API FastAPI — Cyclo v5 avec Rappel Automatique d'Email (2 jours avant les règles)
"""

from fastapi import FastAPI, HTTPException, Header, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from datetime import date, timedelta
from typing import List, Optional
import numpy as np
import os
import requests
from supabase import create_client, Client

app = FastAPI(title="API Prédiction Cycle Menstruel & Automatisations", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_COEFFICIENTS = np.array([0.30558796, 0.3133756, 0.03538975, -0.02215476])
MODEL_INTERCEPT = 11.212948098682684
MODEL_MAE_DAYS = 2.466

SURVIE_SPERMATOZOIDES = 5
SURVIE_OVULE = 1
DEFAULT_LUTEAL_PHASE_DAYS = 14
DUREE_REGLES_MIN = 1
DUREE_REGLES_MAX = 14


def predire_duree_cycle(cycle_precedent, moyenne_3_derniers, ecart_type_3_derniers, cycle_number):
    features = np.array([cycle_precedent, moyenne_3_derniers, ecart_type_3_derniers, cycle_number])
    return float(np.dot(MODEL_COEFFICIENTS, features) + MODEL_INTERCEPT)


# --- SCHÉMAS ---

class CycleEntry(BaseModel):
    date_debut: date = Field(..., description="Premier jour des règles.")
    date_fin: date = Field(..., description="Dernier jour des règles (inclus).")


class PredictionRequest(BaseModel):
    cycles: List[CycleEntry]
    user_email: Optional[EmailStr] = None


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
    compte_a_rebours: str
    en_retard: bool
    phases: List[Phase]
    stats: StatsDashboard
    nombre_cycles_utilises: int
    marge_erreur_jours: float
    explication_marge_erreur: str
    avertissement: str


class ReminderRequest(BaseModel):
    email: EmailStr
    date_prochaines_regles: date


class ReminderResponse(BaseModel):
    success: bool
    message: str


# --- HELPER FUNCTIONS ---

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


def formater_compte_a_rebours(jours_avant: int) -> tuple[str, bool]:
    if jours_avant > 0:
        return f"J-{jours_avant}", False
    elif jours_avant == 0:
        return "Prévues aujourd'hui", False
    else:
        return f"En retard de {abs(jours_avant)} jour{'s' if abs(jours_avant) > 1 else ''}", True


def envoyer_email_resend(email_destinataire: str, date_regles_estimee: date) -> tuple[bool, str]:
    api_key = os.getenv("RESEND_API_KEY", "")

    if not api_key:
        return False, "Cle RESEND_API_KEY non configuree."

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": "Cyclo <onboarding@resend.dev>",
                "to": [email_destinataire],
                "subject": "🌸 Cyclo — Tes règles arrivent dans 2 jours !",
                "text": (
                    f"Bonjour,

"
                    f"Ceci est un petit rappel automatique de Cyclo.
"
                    f"Tes prochaines règles sont estimées pour dans 2 jours, le {date_regles_estimee.strftime('%d/%m/%Y')}.

"
                    f"Prends bien soin de toi !
L'équipe Cyclo"
                ),
            },
            timeout=10,
        )

        if response.status_code in (200, 201):
            return True, "Email envoyé avec succès !"
        else:
            return False, f"Échec de l'envoi (code {response.status_code}) : {response.text}"

    except requests.exceptions.RequestException as e:
        return False, f"Erreur réseau lors de l'envoi : {e}"


# --- ROUTES ---

@app.get("/")
def root():
    return {
        "message": "API Cyclo v5 avec automatisation d'email",
        "version": "5.0",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    cycles = sorted(request.cycles, key=lambda c: c.date_debut)

    if len(cycles) < 2:
        raise HTTPException(status_code=400, detail="Il faut au moins 2 cycles enregistrés.")

    for c in cycles:
        if c.date_fin < c.date_debut:
            raise HTTPException(status_code=400, detail="Date de fin antérieure à la date de début.")

    dates_debut = [c.date_debut for c in cycles]
    durees_cycles = [(dates_debut[i] - dates_debut[i - 1]).days for i in range(1, len(dates_debut))]
    durees_valides = [d for d in durees_cycles if 15 <= d <= 60]

    if not durees_valides:
        raise HTTPException(status_code=400, detail="Durées de cycle invalides.")

    durees_regles_reelles = [(c.date_fin - c.date_debut).days + 1 for c in cycles]
    durees_regles_valides = [d for d in durees_regles_reelles if DUREE_REGLES_MIN <= d <= DUREE_REGLES_MAX]
    moyenne_duree_regles = float(np.mean(durees_regles_valides)) if durees_regles_valides else 5.0

    features = calculer_features_depuis_historique(durees_valides)
    duree_predite = predire_duree_cycle(
        features["cycle_precedent"],
        features["moyenne_3_derniers"],
        features["ecart_type_3_derniers"],
        features["cycle_number"],
    )

    derniere_date_regles = dates_debut[-1]
    date_prochaines_regles = derniere_date_regles + timedelta(days=round(duree_predite))

    jour_ovulation_estime = date_prochaines_regles - timedelta(days=DEFAULT_LUTEAL_PHASE_DAYS)
    date_debut_ovulation = jour_ovulation_estime - timedelta(days=2)
    date_fin_ovulation = jour_ovulation_estime + timedelta(days=2)

    phases = construire_phases(
        derniere_date_regles=derniere_date_regles,
        date_prochaines_regles=date_prochaines_regles,
        jour_ovulation_estime=jour_ovulation_estime,
        duree_regles=round(moyenne_duree_regles),
    )

    aujourdhui = date.today()
    jour_actuel_cycle = (aujourdhui - derniere_date_regles).days + 1
    jours_avant_prochaines_regles = (date_prochaines_regles - aujourdhui).days
    compte_a_rebours, en_retard = formater_compte_a_rebours(jours_avant_prochaines_regles)

    phase_actuelle = phases[0]
    for p in phases:
        if p.date_debut <= aujourdhui <= p.date_fin:
            phase_actuelle = p
            break
    else:
        if aujourdhui > phases[-1].date_fin:
            phase_actuelle = Phase(
                nom="Retard potentiel",
                date_debut=date_prochaines_regles,
                date_fin=aujourdhui,
                niveau_fertilite="faible",
                conseil="Prochaines règles en retard par rapport à la prédiction moyenne.",
            )

    stats = StatsDashboard(
        moyenne_duree_cycle=round(float(np.mean(durees_valides)), 1),
        moyenne_duree_regles=round(moyenne_duree_regles, 1),
        ecart_type_cycle=round(float(np.std(durees_valides)), 1),
        nombre_cycles_enregistres=len(durees_valides),
    )

    explication_marge = f"Variation moyenne de {MODEL_MAE_DAYS} jours sur les cycles de test."
    avertissement = "Estimation informative uniquement. Ne constitue pas un moyen contraceptif."

    return PredictionResponse(
        duree_cycle_predite=round(duree_predite, 1),
        date_prochaines_regles=date_prochaines_regles,
        date_debut_ovulation=date_debut_ovulation,
        date_fin_ovulation=date_fin_ovulation,
        jour_actuel_cycle=jour_actuel_cycle,
        phase_actuelle=phase_actuelle,
        jours_avant_prochaines_regles=jours_avant_prochaines_regles,
        compte_a_rebours=compte_a_rebours,
        en_retard=en_retard,
        phases=phases,
        stats=stats,
        nombre_cycles_utilises=len(durees_valides),
        marge_erreur_jours=MODEL_MAE_DAYS,
        explication_marge_erreur=explication_marge,
        avertissement=avertissement,
    )


@app.post("/send-reminder", response_model=ReminderResponse)
def send_reminder(request: ReminderRequest):
    success, message = envoyer_email_resend(request.email, request.date_prochaines_regles)
    if not success:
        raise HTTPException(status_code=500, detail=message)
    return ReminderResponse(success=True, message=message)


@app.get("/cron/send-reminders")
def cron_send_reminders(authorization: Optional[str] = Header(None)):
    cron_secret = os.getenv("CRON_SECRET", "")
    if cron_secret and authorization != f"Bearer {cron_secret}":
        raise HTTPException(status_code=401, detail="Non autorisé.")

    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    if not supabase_url or not supabase_service_key:
        raise HTTPException(status_code=500, detail="Variables Supabase manquantes.")

    supabase: Client = create_client(supabase_url, supabase_service_key)

    # Récupérer tous les utilisateurs
    users_resp = supabase.auth.admin.list_users()
    users = users_resp if isinstance(users_resp, list) else getattr(users_resp, "users", [])

    cibles_notifiees = []
    aujourdhui = date.today()

    for u in users:
        user_id = u.id
        user_email = u.email

        cycles_resp = supabase.table("cycles").select("date_debut_regles, date_fin_regles").eq("user_id", user_id).order("date_debut_regles", desc=False).execute()
        cycles_data = cycles_resp.data

        if not cycles_data or len(cycles_data) < 2:
            continue

        entries = [
            CycleEntry(date_debut=c["date_debut_regles"], date_fin=c["date_fin_regles"])
            for c in cycles_data if c.get("date_fin_regles")
        ]

        if len(entries) < 2:
            continue

        dates_debut = [c.date_debut for c in entries]
        durees_cycles = [(dates_debut[i] - dates_debut[i - 1]).days for i in range(1, len(dates_debut))]
        durees_valides = [d for d in durees_cycles if 15 <= d <= 60]

        if not durees_valides:
            continue

        features = calculer_features_depuis_historique(durees_valides)
        duree_predite = predire_duree_cycle(
            features["cycle_precedent"],
            features["moyenne_3_derniers"],
            features["ecart_type_3_derniers"],
            features["cycle_number"],
        )

        derniere_date = dates_debut[-1]
        date_prochaines = derniere_date + timedelta(days=round(duree_predite))

        # Vérifier si l'échéance est exactement dans 2 jours
        if (date_prochaines - aujourdhui).days == 2:
            ok, msg = envoyer_email_resend(user_email, date_prochaines)
            if ok:
                cibles_notifiees.append(user_email)

    return {"status": "ok", "emails_envoyes": cibles_notifiees}
