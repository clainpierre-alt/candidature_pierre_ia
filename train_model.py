import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# DATASET HYBRIDE : SAV SPORT DÉCOUVERTE + JUMEAU NUMÉRIQUE PIERRE CLAIN
data = [
    # ==========================================
    # PARTIE 1 : JUMEAU NUMÉRIQUE PIERRE CLAIN
    # ==========================================
    # --- TWIN : IDENTITÉ & CV ---
    {"text": "Qui est Pierre Clain ?", "intent": "twin_identity"},
    {"text": "Présente-toi", "intent": "twin_identity"},
    {"text": "Parle-moi de ton parcours professionnel", "intent": "twin_identity"},
    {"text": "Quelle est ta date de naissance et ton origine ?", "intent": "twin_identity"},
    {"text": "Où es-tu né et quel âge as-tu ?", "intent": "twin_identity"},
    {"text": "Fais-moi un résumé de ton CV", "intent": "twin_identity"},

    # --- TWIN : FORMATION & DATABIRD ---
    {"text": "Quelles sont tes certifications et formations Data ?", "intent": "twin_education"},
    {"text": "Parle-moi de ta formation Data Analyst chez DataBird", "intent": "twin_education"},
    {"text": "Quels diplômes as-tu obtenus ?", "intent": "twin_education"},

    # --- TWIN : DECATHLON & P&L ---
    {"text": "Raconte-moi ton expérience chez Decathlon", "intent": "twin_decathlon"},
    {"text": "Quel était ton rôle en tant que Responsable Exploitation ?", "intent": "twin_decathlon"},
    {"text": "Comment gérais-tu le P&L et les équipes chez Decathlon ?", "intent": "twin_decathlon"},
    {"text": "As-tu de l'expérience en gestion de magasin et service client ?", "intent": "twin_decathlon"},

    # --- TWIN : TECH & STACK DATA ---
    {"text": "Quelles sont tes compétences techniques en Data ?", "intent": "twin_tech"},
    {"text": "Maîtrises-tu Python, SQL, Pandas et Power BI ?", "intent": "twin_tech"},
    {"text": "As-tu des compétences en Machine Learning et NLP ?", "intent": "twin_tech"},

    # --- TWIN : PROJET SOOBIK ---
    {"text": "C'est quoi le projet Soobik ?", "intent": "twin_soobik"},
    {"text": "Peux-tu me parler de ton projet de location de matériel à La Réunion ?", "intent": "twin_soobik"},

    # --- TWIN : MOTIVATION SPORT DÉCOUVERTE ---
    {"text": "Pourquoi postules-tu chez Sport Découverte ?", "intent": "twin_motivation"},
    {"text": "Pourquoi toi pour le poste de Responsable Partenariats ?", "intent": "twin_motivation"},
    {"text": "Qu'apportes-tu au réseau de partenaires de Sport Découverte ?", "intent": "twin_motivation"},


    # ==========================================
    # PARTIE 2 : SAV SPORT DÉCOUVERTE (ANCIENNES RÉPONSES CONSERVÉES)
    # ==========================================
    # --- SAV : MÉTÉO ---
    {"text": "Saut en parachute annulé cause météo", "intent": "sav_meteo"},
    {"text": "Il pleut trop pour mon vol en montgolfière", "intent": "sav_meteo"},
    {"text": "Le vent est trop fort, activité reportée", "intent": "sav_meteo"},
    {"text": "Vol annulé à cause de la pluie et de l'orage", "intent": "sav_meteo"},
    {"text": "Mon activité extérieure est-elle maintenue sous la pluie ?", "intent": "sav_meteo"},
    {"text": "Report météo partenaire vol hélicoptère", "intent": "sav_meteo"},

    # --- SAV : PARTENAIRE INJOIGNABLE ---
    {"text": "Le partenaire ne répond pas au téléphone", "intent": "sav_injoignable"},
    {"text": "Impossible de joindre le prestataire", "intent": "sav_injoignable"},
    {"text": "Le centre de saut ne décroche pas", "intent": "sav_injoignable"},
    {"text": "Aucune réponse à mes appels au centre de pilotage", "intent": "sav_injoignable"},
    {"text": "J'essaie de contacter le club mais personne ne répond", "intent": "sav_injoignable"},

    # --- SAV : REPORT / CHANGEMENT DE DATE ---
    {"text": "Je veux décaler la date de ma réservation", "intent": "sav_report"},
    {"text": "Est-ce possible de changer de créneau horaire ?", "intent": "sav_report"},
    {"text": "Je souhaite modifier le jour de mon stage de conduite", "intent": "sav_report"},
    {"text": "Comment changer la date de mon bon cadeau ?", "intent": "sav_report"},

    # --- SAV : REMBOURSEMENT & AVOIR ---
    {"text": "Je demande le remboursement intégral", "intent": "sav_remboursement"},
    {"text": "Je veux annuler ma commande et avoir un virement", "intent": "sav_remboursement"},
    {"text": "Remboursez-moi mon bon cadeau expiré", "intent": "sav_remboursement"},
    {"text": "Comment se faire rembourser une prestation ?", "intent": "sav_remboursement"}
]

df = pd.DataFrame(data)

# Pipeline de classification
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, lowercase=True)),
    ('clf', LogisticRegression(C=3.0, max_iter=300))
])

print("⚙️ Entraînement du modèle hybride (Jumeau + SAV)...")
pipeline.fit(df['text'], df['intent'])

os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/intent_classifier.joblib')
print("✅ Modèle hybride sauvegardé avec succès dans 'model/intent_classifier.joblib'")
