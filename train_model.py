import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# DATASET ÉTENDU (+100 PHRASES)
data = [
    # ==================== 1. SAV : MÉTÉO / ANNULATION ====================
    {"text": "Saut en parachute annulé cause météo", "intent": "sav_meteo"},
    {"text": "Il pleut trop pour mon vol en montgolfière", "intent": "sav_meteo"},
    {"text": "Le vent est trop fort, activité reportée", "intent": "sav_meteo"},
    {"text": "Vol annulé à cause de la pluie et de l'orage", "intent": "sav_meteo"},
    {"text": "Mon activité extérieure est-elle maintenue sous la pluie ?", "intent": "sav_meteo"},
    {"text": "Report météo partenaire vol hélicoptère", "intent": "sav_meteo"},
    {"text": "Conditions météorologiques défavorables", "intent": "sav_meteo"},
    {"text": "Puis-je annuler si la météo est mauvaise ?", "intent": "sav_meteo"},
    {"text": "Le moniteur a annulé le saut à cause du vent", "intent": "sav_meteo"},
    {"text": "Est-ce qu'on vole s'il y a du brouillard ?", "intent": "sav_meteo"},
    {"text": "Tempête de neige stage de pilotage fermé", "intent": "sav_meteo"},
    {"text": "Météo défavorable pour le baptême de l'air", "intent": "sav_meteo"},
    {"text": "Activité reportée cause mauvais temps", "intent": "sav_meteo"},
    {"text": "Prévisions météo mauvaises pour ce week-end", "intent": "sav_meteo"},

    # ==================== 2. SAV : PARTENAIRE INJOIGNABLE ====================
    {"text": "Le partenaire ne répond pas au téléphone", "intent": "sav_injoignable"},
    {"text": "Impossible de joindre le prestataire", "intent": "sav_injoignable"},
    {"text": "Le centre de saut ne décroche pas", "intent": "sav_injoignable"},
    {"text": "Aucune réponse à mes appels au centre de pilotage", "intent": "sav_injoignable"},
    {"text": "J'essaie de contacter le club mais personne ne répond", "intent": "sav_injoignable"},
    {"text": "Numéro de téléphone du partenaire injoignable", "intent": "sav_injoignable"},
    {"text": "Prestataire absent sur la messagerie", "intent": "sav_injoignable"},
    {"text": "Je n'arrive pas à fixer un rendez-vous avec l'organisateur", "intent": "sav_injoignable"},
    {"text": "Le téléphone sonne dans le vide chez le partenaire", "intent": "sav_injoignable"},
    {"text": "Pas de nouvelles du club de plongée", "intent": "sav_injoignable"},
    {"text": "L'école de pilotage ne rappelle pas", "intent": "sav_injoignable"},

    # ==================== 3. SAV : REPORT / CHANGEMENT DE DATE ====================
    {"text": "Je veux décaler la date de ma réservation", "intent": "sav_report"},
    {"text": "Est-ce possible de changer de créneau horaire ?", "intent": "sav_report"},
    {"text": "Je souhaite modifier le jour de mon stage de conduite", "intent": "sav_report"},
    {"text": "Comment changer la date de mon bon cadeau ?", "intent": "sav_report"},
    {"text": "Est-ce possible de repousser la réservation à la semaine prochaine ?", "intent": "sav_report"},
    {"text": "Changement de date sans frais", "intent": "sav_report"},
    {"text": "Je peux reporter mon vol en hélico ?", "intent": "sav_report"},
    {"text": "Décaler le rendez-vous au mois prochain", "intent": "sav_report"},
    {"text": "Comment modifier mon horaire de passage ?", "intent": "sav_report"},
    {"text": "Changer le nom du bénéficiaire du bon", "intent": "sav_report"},

    # ==================== 4. SAV : REMBOURSEMENT & AVOIR ====================
    {"text": "Je demande le remboursement intégral", "intent": "sav_remboursement"},
    {"text": "Je veux annuler ma commande et avoir un virement", "intent": "sav_remboursement"},
    {"text": "Remboursez-moi mon bon cadeau expiré", "intent": "sav_remboursement"},
    {"text": "Je souhaite obtenir un virement de remboursement", "intent": "sav_remboursement"},
    {"text": "Comment se faire rembourser une prestation ?", "intent": "sav_remboursement"},
    {"text": "Procédure d'annulation et remboursement", "intent": "sav_remboursement"},
    {"text": "Rembourser sur ma carte bancaire s'il vous plaît", "intent": "sav_remboursement"},
    {"text": "Je souhaite convertir mon achat en avoir", "intent": "sav_remboursement"},
    {"text": "Modalités d'avoir suite annulation", "intent": "sav_remboursement"},

    # ==================== 5. JUMEAU : P&L, MANAGEMENT & EXPLOITATION ====================
    {"text": "Quel est ton bilan managérial et P&L ?", "intent": "twin_pnl"},
    {"text": "Comment pilotes-tu le compte de résultat ?", "intent": "twin_pnl"},
    {"text": "Quelle est ton expérience en gestion de budget et magasin ?", "intent": "twin_pnl"},
    {"text": "Raconte-moi tes résultats chez Decathlon", "intent": "twin_pnl"},
    {"text": "As-tu déjà géré une équipe et un compte d'exploitation ?", "intent": "twin_pnl"},
    {"text": "Comment gères-tu la rentabilité et le chiffre d'affaires ?", "intent": "twin_pnl"},
    {"text": "Pilotage de la masse salariale et rentabilité", "intent": "twin_pnl"},
    {"text": "Quelle est ta méthode de management d'équipe ?", "intent": "twin_pnl"},
    {"text": "Responsable exploitation et service client", "intent": "twin_pnl"},
    {"text": "Gestion des stocks et marge commerciale", "intent": "twin_pnl"},

    # ==================== 6. JUMEAU : DATA, SQL, PYTHON & ML ====================
    {"text": "Quelles sont tes compétences en Python, SQL et Data ?", "intent": "twin_data"},
    {"text": "Parle-moi de ta certification Data Analyse", "intent": "twin_data"},
    {"text": "Comment utilises-tu Power BI et Pandas ?", "intent": "twin_data"},
    {"text": "Que maîtrises-tu en analyse de données et modèles ML ?", "intent": "twin_data"},
    {"text": "As-tu des projets en SQL, Python ou modélisation ?", "intent": "twin_data"},
    {"text": "Explique-moi tes compétences techniques Data", "intent": "twin_data"},
    {"text": "Certification DataBird et projets accomplis", "intent": "twin_data"},
    {"text": "Connais-tu Scikit-Learn et l'analyse prédictive ?", "intent": "twin_data"},
    {"text": "Création de tableaux de bord PowerBI et KPI", "intent": "twin_data"},
    {"text": "Nettoyage et traitement de données avec Pandas", "intent": "twin_data"},

    # ==================== 7. JUMEAU : MOTIVATION & SPORT DÉCOUVERTE ====================
    {"text": "Pourquoi postules-tu chez Sport Découverte ?", "intent": "twin_motivation"},
    {"text": "Quelle est ta vision pour le réseau de partenaires ?", "intent": "twin_motivation"},
    {"text": "Pourquoi devrions-nous te recruter ?", "intent": "twin_motivation"},
    {"text": "Qu'apportes-tu à l'équipe de Sport Découverte ?", "intent": "twin_motivation"},
    {"text": "Quelle est ta motivation pour le secteur de l'outdoor et des loisirs ?", "intent": "twin_motivation"},
    {"text": "Pourquoi le poste de Responsable Partenariats ?", "intent": "twin_motivation"},
    {"text": "Comment comptes-tu développer le réseau de partenaires outdoor ?", "intent": "twin_motivation"}
]

df = pd.DataFrame(data)

# Pipeline de vectorisation avancée (TF-IDF sublinéaire sur n-grammes 1 à 3)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3), # Détecte les associations de 1, 2 ou 3 mots
        sublinear_tf=True,  # Évite qu'un mot répétitif écrase le sens
        lowercase=True
    )),
    ('clf', LogisticRegression(C=3.0, max_iter=300))
])

print("⚙️ Entraînement du modèle avec dataset étendu (+100 exemples)...")
pipeline.fit(df['text'], df['intent'])

os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/intent_classifier.joblib')
print("✅ Modèle sauvegardé avec succès dans 'model/intent_classifier.joblib'")
