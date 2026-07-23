import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Base de connaissances (Intention -> Exemples de phrases)
data = [
    {"text": "Saut en parachute annulé cause météo", "intent": "sav_meteo"},
    {"text": "Il pleut trop pour mon vol en montgolfière", "intent": "sav_meteo"},
    {"text": "Le vent est trop fort, activité reportée", "intent": "sav_meteo"},
    {"text": "Le partenaire ne répond pas au téléphone", "intent": "sav_injoignable"},
    {"text": "Impossible de joindre le prestataire", "intent": "sav_injoignable"},
    {"text": "Je veux décaler la date de ma réservation", "intent": "sav_report"},
    {"text": "Est-ce possible de changer de créneau horraire ?", "intent": "sav_report"},
    {"text": "Je demande le remboursement intégral", "intent": "sav_remboursement"},
    {"text": "Je veux annuler ma commande et avoir un virement", "intent": "sav_remboursement"},
    {"text": "Quel est ton bilan managérial et P&L ?", "intent": "twin_pnl"},
    {"text": "Comment pilotes-tu le compte de résultat ?", "intent": "twin_pnl"},
    {"text": "Quelles sont tes compétences en Python, SQL et Data ?", "intent": "twin_data"},
    {"text": "Parle-moi de ta certification Data Analyse", "intent": "twin_data"},
    {"text": "Pourquoi postules-tu chez Sport Découverte ?", "intent": "twin_motivation"},
    {"text": "Quelle est ta vision pour le réseau de partenaires ?", "intent": "twin_motivation"}
]

df = pd.DataFrame(data)

# 2. Pipeline ML (TF-IDF + Régression Logistique)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
    ('clf', LogisticRegression(C=1.0, max_iter=200))
])

# 3. Entraînement
print("⚙️ Entraînement du modèle ML en cours...")
pipeline.fit(df['text'], df['intent'])

# 4. Sauvegarde
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/intent_classifier.joblib')
print("✅ Modèle ML sauvegardé avec succès dans 'model/intent_classifier.joblib'")
