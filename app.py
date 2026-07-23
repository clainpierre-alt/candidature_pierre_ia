import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="API Candidature & SAV - Pierre Clain",
    description="API hybride de classification NLP (SAV Sport Découverte + Jumeau Numérique Pierre Clain)",
    version="1.0.0"
)

MODEL_PATH = "model/intent_classifier.joblib"

# ------------------------------------------------------------------------------
# CHARGEMENT / ENTRAÎNEMENT AUTOMATIQUE DU MODÈLE
# ------------------------------------------------------------------------------
def load_or_train_model():
    """Charge le modèle ML s'il existe, sinon exécute train_model.py pour le générer."""
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Modèle non trouvé au démarrage. Lancement automatique de train_model.py...")
        exit_code = os.system("python train_model.py")
        if exit_code != 0:
            print("❌ Erreur lors de l'exécution de train_model.py")
            return None
    try:
        loaded_model = joblib.load(MODEL_PATH)
        print("✅ Modèle chargé avec succès !")
        return loaded_model
    except Exception as e:
        print(f"❌ Impossible de charger le modèle : {e}")
        return None

model = load_or_train_model()

# ------------------------------------------------------------------------------
# BASE DE RÉPONSES DYNAMIQUE (JUMEAU NUMÉRIQUE + SAV)
# ------------------------------------------------------------------------------
RESPONSES = {
    # ==================== BLOC 1 : JUMEAU NUMÉRIQUE ====================
    "twin_identity": (
        "Je suis Pierre CLAIN, né le 03/07/1994 à Melun. Manager d'exploitation et de service client expérimenté chez Decathlon, "
        "j'ai complété mon profil par une expertise en Data Analyse (certification DataBird). "
        "Je combine la réalité du terrain commerçant avec la puissance des modèles d'analyse de données."
    ),
    "twin_education": (
        "Je suis certifié Data Analyst par l'organisme DataBird (validation en mars 2026). "
        "Ma formation couvre le traitement complet de la donnée : requêtage SQL avancé, nettoyage Python (Pandas/NumPy), "
        "modélisation Machine Learning (Scikit-Learn) et création de tableaux de bord interactifs sur Power BI."
    ),
    "twin_decathlon": (
        "En tant que Responsable Exploitation et Service Client chez Decathlon, j'ai piloté l'animation commerciale, "
        "le management d'équipes pluridisciplinaires, la gestion de la masse salariale et le compte de résultat (P&L), "
        "tout en garantissant une excellente satisfaction client sur le terrain."
    ),
    "twin_tech": (
        "Mon stack technique comprend : Python (Pandas, Scikit-Learn, FastAPI), SQL, Power BI, DAX, "
        "et l'intégration d'automatisation via Google Apps Script et APIs REST."
    ),
    "twin_soobik": (
        "Soobik est un projet d'entreprise que j'ai modélisé pour La Réunion : un service de location de matériel de vacances et de trek, "
        "avec livraison sur lieu de séjour et aéroport. J'y ai conçu l'étude de marché, le business plan P&L et la modélisation du besoin client."
    ),
    "twin_motivation": (
        "Ma double culture (Commerçant & Data Analyst) me permet de comprendre les enjeux opérationnels des partenaires outdoor de Sport Découverte "
        "tout en déployant des outils d'analyse de données et d'automatisation pour maximiser la rentabilité du réseau."
    ),

    # ==================== BLOC 2 : SAV SPORT DÉCOUVERTE ====================
    "sav_meteo": (
        "En cas de météo défavorable (pluie, vent fort, orage), la prestation est suspendue pour votre sécurité.\n\n"
        "Votre bon reste 100% valide. Vous pouvez reprogrammer une nouvelle date sans aucun frais depuis votre espace client."
    ),
    "sav_injoignable": (
        "Nos partenaires sont régulièrement en prestation sur le terrain.\n\n"
        "Nous venons de leur transmettre une alerte prioritaire afin qu'ils vous recontactent sous 4 heures ouvrées."
    ),
    "sav_report": (
        "Vous pouvez modifier gratuitement la date de votre activité depuis votre espace 'Mes Réservations' jusqu'à 48 heures avant le rendez-vous."
    ),
    "sav_remboursement": (
        "En cas d'annulation confirmée par le prestataire ou de cas de force majeure, le remboursement par virement est traité sous 3 à 5 jours ouvrés."
    )
}

# ------------------------------------------------------------------------------
# DÉFINITION DE LA REQUÊTE ET DES ENDPOINTS
# ------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "API Candidature Pierre Clain & SAV opérationnelle.",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_intent(request: QueryRequest):
    global model
    if model is None:
        # Tentative de rechargement au cas où le modèle aurait été généré entre-temps
        model = load_or_train_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Modèle non disponible sur le serveur.")

    clean_text = request.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    # Prédiction de l'intention et calcul des probabilités
    predicted_intent = model.predict([clean_text])[0]
    probabilities = model.predict_proba([clean_text])[0]
    confidence = float(max(probabilities))

    # Gestion du seuil de confiance (Seuil ajusté à 0.22 pour assurer une flexibilité)
    if confidence < 0.22:
        reply = (
            "Votre demande spécifique nécessite l'intervention d'un conseiller. "
            "Un membre de l'équipe prend le relais pour vous répondre sous peu."
        )
    else:
        reply = RESPONSES.get(
            predicted_intent, 
            "Un conseiller analyse votre demande et revient vers vous rapidement."
        )

    return {
        "success": True,
        "query": clean_text,
        "intent": predicted_intent,
        "confidence": round(confidence, 2),
        "reply": reply
    }
