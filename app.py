from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Pierre Clain - Machine Learning Intent Engine")

# Charger le modèle Scikit-Learn
MODEL_PATH = "model/intent_classifier.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Base de réponses humaines et ultra-détaillées associées aux intentions ML
RESPONSES_DATABASE = {
    "sav_meteo": (
        "Bonjour ! Je comprends tout à fait votre déception. Une annulation météo reste frustrante mais la sécurité est primordiale.\n\n"
        "Votre bon a été réactivé dans votre espace. Vous pouvez :\n"
        "1. Reprogrammer une nouvelle date sans frais.\n"
        "2. Convertir votre bon en avoir sur l'ensemble du catalogue Sport Découverte."
    ),
    "sav_injoignable": (
        "Bonjour, navré pour ce désagrément. Nos partenaires sont souvent en pleine prestation sur le terrain.\n\n"
        "Nous venons d'envoyer une alerte prioritaire au responsable de la structure pour qu'il vous rappelle directement sous 4h."
    ),
    "sav_report": (
        "Bonjour ! Vous pouvez décaler gratuitement votre réservation depuis votre espace 'Mes Réservations' si vous êtes à plus de 72h du rendez-vous."
    ),
    "sav_remboursement": (
        "Bonjour, si l'annulation fait suite à une décision de la structure ou d'un cas de force majeure, votre remboursement bancaire est effectué sous 3 à 5 jours ouvrés."
    ),
    "twin_pnl": (
        "Ancien Responsable Exploitation & Service Client chez Decathlon, j'ai piloté l'animation d'équipes pluridisciplinaires, la performance commerciale et la gestion directe du P&L."
    ),
    "twin_data": (
        "Certifié Data Analyst chez DataBird, je maîtrise SQL, Python (Pandas/Scikit-Learn), Power BI et l'automatisation de workflows (API/Webhooks)."
    ),
    "twin_motivation": (
        "Mon ambition chez Sport Découverte est de conjuguer mon expérience de terrain commerçante à la puissance de la Data pour développer le réseau de partenaires."
    )
}

class QueryRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_intent(request: QueryRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texte vide")

    # Prédiction ML
    predicted_intent = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities))

    # Récupération de la réponse si la confiance est jugée suffisante
    if confidence < 0.22:
        reply = "Bonjour ! J'ai bien pris note de votre message. Un conseiller dédié va analyser votre demande spécifique et vous répondre sous 4 heures."
    else:
        reply = RESPONSES_DATABASE.get(predicted_intent, "Bonjour, un conseiller étudie votre demande.")

    return {
        "success": True,
        "intent": predicted_intent,
        "confidence": round(confidence, 2),
        "reply": reply
    }
