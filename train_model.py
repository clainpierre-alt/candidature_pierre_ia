import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =====================================================================
# 1. JEU DE DONNÉES DE DÉMONSTRATION / APPRENTISSAGE NLP SAV & JUMEAU
# =====================================================================
data = [
    # --- INTENTION : SAV_METEO ---
    {"text": "Saut en parachute annulé cause météo", "intent": "sav_meteo"},
    {"text": "Il pleut trop pour mon vol en montgolfière", "intent": "sav_meteo"},
    {"text": "Le vent est trop fort, activité reportée", "intent": "sav_meteo"},
    {"text": "Mon stage de pilotage est décalé pour mauvaise météo", "intent": "sav_meteo"},

    # --- INTENTION : SAV_PARTENAIRE_INJOIGNABLE ---
    {"text": "Le partenaire ne répond pas au téléphone", "intent": "sav_injoignable"},
    {"text": "Impossible de joindre le prestataire pour valider mon bon", "intent": "sav_injoignable"},
    {"text": "Personne au bout du fil, ligne occupée depuis ce matin", "intent": "sav_injoignable"},
    {"text": "J'essaie d'appeler l'organisateur mais pas de réponse", "intent": "sav_injoignable"},

    # --- INTENTION : SAV_REPORT_DATE ---
    {"text": "Je veux décaler la date de ma réservation", "intent": "sav_report"},
    {"text": "Est-ce possible de changer de créneau horraire ?", "intent": "sav_report"},
    {"text": "J'ai un empêchement ce week-end pour mon activité", "intent": "sav_report"},
    {"text": "Reporter mon bon cadeaux à un autre mois", "intent": "sav_report"},

    # --- INTENTION : SAV_REMBOURSEMENT ---
    {"text": "Je demande le remboursement intégral", "intent": "sav_remboursement"},
    {"text": "Comment récupérer mon argent suite annulation ?", "intent": "sav_remboursement"},
    {"text": "Je veux annuler ma commande et avoir un virement", "intent": "sav_remboursement"},

    # --- INTENTION : TWIN_EXPERIENCE_MANAGEMENT ---
    {"text": "Quel est ton bilan managérial et P&L ?", "intent": "twin_pnl"},
    {"text": "As-tu déjà géré une équipe chez Decathlon ?", "intent": "twin_pnl"},
    {"text": "Comment pilotes-tu le compte de résultat et le budget ?", "intent": "twin_pnl"},
    {"text": "Raconte-moi ton expérience en magasin", "intent": "twin_pnl"},

    # --- INTENTION : TWIN_DATA_TECH ---
    {"text": "Quelles sont tes compétences en Python, SQL et Data ?", "intent": "twin_data"},
    {"text": "Parle-moi de ta certification Data Analyse", "intent": "twin_data"},
    {"text": "Manières-tu Power BI, Pandas et le Machine Learning ?", "intent": "twin_data"},
    {"text": "Peux-tu construire des pipelines d'automatisation ?", "intent": "twin_data"},

    # --- INTENTION : TWIN_MOTIVATION_SPORTDECOUVERTE ---
    {"text": "Pourquoi postules-tu chez Sport Découverte ?", "intent": "twin_motivation"},
    {"text": "Quelle est ta vision pour le réseau de partenaires ?", "intent": "twin_motivation"},
    {"text": "Pourquoi te choisir comme Responsable Partenariats ?", "intent": "twin_motivation"}
]

# Convertir en DataFrame Pandas
df = pd.DataFrame(data)

# =====================================================================
# 2. PIPELINE MACHINE LEARNING (NLP TF-IDF + LOGISTIC REGRESSION)
# =====================================================================
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
    ('clf', LogisticRegression(C=1.0, max_iter=200))
])

# Entraînement
X = df['text']
y = df['intent']
pipeline.fit(X, y)

# =====================================================================
# 3. SAUVEGARDE DU MODÈLE POUR PRODUCTION (.joblib)
# =====================================================================
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/intent_classifier.joblib')
print("✅ Modèle Machine Learning scikit-learn entraîné et sauvegardé sous 'model/intent_classifier.joblib' !")
