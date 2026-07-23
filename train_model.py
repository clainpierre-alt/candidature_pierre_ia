import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Base d'apprentissage ultra-enrichie (Data Augmentation)
data = [
    # --- INTENT 1: SAV MÉTÉO ---
    {"text": "Saut en parachute annulé cause météo", "intent": "sav_meteo"},
    {"text": "Il pleut trop pour mon vol en montgolfière", "intent": "sav_meteo"},
    {"text": "Le vent est trop fort, activité reportée", "intent": "sav_meteo"},
    {"text": "Vol annulé à cause de la pluie et de l'orage", "intent": "sav_meteo"},
    {"text": "Mon activité extérieure est-elle maintenue sous la pluie ?", "intent": "sav_meteo"},
    {"text": "Report météo partenaire vol hélicoptère", "intent": "sav_meteo"},
    {"text": "Conditions météorologiques défavorables", "intent": "sav_meteo"},
    {"text": "Puis-je annuler si la météo est mauvaise ?", "intent": "sav_meteo"},

    # --- INTENT 2: SAV PARTENAIRE INJOIGNABLE ---
    {"text": "Le partenaire ne répond pas au téléphone", "intent": "sav_injoignable"},
    {"text": "Impossible de joindre le prestataire", "intent": "sav_injoignable"},
    {"text": "Le centre de saut ne décroche pas", "intent": "sav_injoignable"},
    {"text": "Aucune réponse à mes appels au centre de pilotage", "intent": "sav_injoignable"},
    {"text": "J'essaie de contacter le club mais personne ne répond", "intent": "sav_injoignable"},
    {"text": "Numéro de téléphone du partenaire injoignable", "intent": "sav_injoignable"},

    # --- INTENT 3: SAV REPORT / MODIFICATION ---
    {"text": "Je veux décaler la date de ma réservation", "intent": "sav_report"},
    {"text": "Est-ce possible de changer de créneau horaire ?", "intent": "sav_report"},
    {"text": "Je souhaite modifier le jour de mon stage de conduite", "intent": "sav_report"},
    {"text": "Comment changer la date de mon bon cadeau ?", "intent": "sav_report"},
    {"text": "Est-ce possible de repousser la réservation à la semaine prochaine ?", "intent": "sav_report"},
    {"text": "Changement de date sans frais", "intent": "sav_report"},

    # --- INTENT 4: SAV REMBOURSEMENT / AVOIR ---
    {"text": "Je demande le remboursement intégral", "intent": "sav_remboursement"},
    {"text": "Je veux annuler ma commande et avoir un virement", "intent": "sav_remboursement"},
    {"text": "Remboursez-moi mon bon cadeau expiré", "intent": "sav_remboursement"},
    {"text": "Je souhaite obtenir un virement de remboursement", "intent": "sav_remboursement"},
    {"text": "Comment se faire rembourser une prestation ?", "intent": "sav_remboursement"},
    {"text": "Procédure d'annulation et remboursement", "intent": "sav_remboursement"},

    # --- INTENT 5: TWIN P&L & MANAGEMENT ---
    {"text": "Quel est ton bilan managérial et P&L ?", "intent": "twin_pnl"},
    {"text": "Comment pilotes-tu le compte de résultat ?", "intent": "twin_pnl"},
    {"text": "Quelle est ton expérience en gestion de budget et magasin ?", "intent": "twin_pnl"},
    {"text": "Raconte-moi tes résultats chez Decathlon", "intent": "twin_pnl"},
    {"text": "As-tu déjà géré une équipe et un compte d'exploitation ?", "intent": "twin_pnl"},
    {"text": "Comment gères-tu la rentabilité et le chiffre d'affaires ?", "intent": "twin_pnl"},

    # --- INTENT 6: TWIN DATA & COMPÉTENCES ---
    {"text": "Quelles sont tes compétences en Python, SQL et Data ?", "intent": "twin_data"},
    {"text": "Parle-moi de ta certification Data Analyse", "intent": "twin_data"},
    {"text": "Comment utilises-tu Power BI et Pandas ?", "intent": "twin_data"},
    {"text": "Que maîtrises-tu en analyse de données et modèles ML ?", "intent": "twin_data"},
    {"text": "As-tu des projets en SQL, Python ou modélisation ?", "intent": "twin_data"},
    {"text": "Explique-moi tes compétences techniques Data", "intent": "twin_data"},

    # --- INTENT 7: TWIN MOTIVATION & SPORT DÉCOUVERTE ---
    {"text": "Pourquoi postules-tu chez Sport Découverte ?", "intent": "twin_motivation"},
    {"text": "Quelle est ta vision pour le réseau de partenaires ?", "intent": "twin_motivation"},
    {"text": "Pourquoi devrions-nous te recruter ?", "intent": "twin_motivation"},
    {"text": "Qu'apportes-tu à l'équipe de Sport Découverte ?", "intent": "twin_motivation"},
    {"text": "Quelle est ta motivation pour le secteur de l'outdoor et des loisirs ?", "intent": "twin_motivation"}
]

df = pd.DataFrame(data)

# Pipeline de classification NLP
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3), # Prend en compte les mots uniques, paires et triplets de mots
        sublinear_tf=True,  # Atténue l'impact des mots trop fréquents
        min_df=1, 
        lowercase=True
    )),
    ('clf', LogisticRegression(C=5.0, max_iter=300)) # Augmentation de C pour un meilleur apprentissage
])

print("⚙️ Ré-entraînement du modèle avec dataset élargi...")
pipeline.fit(df['text'], df['intent'])

os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/intent_classifier.joblib')
print("✅ Modèle ré-entraîné et sauvegardé avec succès dans 'model/intent_classifier.joblib'")
