const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Configuration des URLs de services
const APPS_SCRIPT_URL = process.env.APPS_SCRIPT_URL;
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';

// --- HELPERS COMMUNICATION ---

// 1. Interrogation du micro-service Python FastAPI (Scikit-Learn)
async function callFastAPI(text) {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000); // Timeout 3s

        const response = await fetch(`${FASTAPI_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (!response.ok) return null;
        return await response.json();
    } catch (error) {
        console.warn("⚠️ API FastAPI indisponible ou délai dépassé. Bascule sur Apps Script...");
        return null; // Déclenche le fallback vers Apps Script
    }
}

// 2. Interrogation centralisée Google Apps Script
async function callAppsScript(action, payload) {
    if (!APPS_SCRIPT_URL) {
        throw new Error("Variable APPS_SCRIPT_URL non configurée dans les variables d'environnement.");
    }
    
    const response = await fetch(APPS_SCRIPT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...payload }),
        redirect: 'follow'
    });

    if (!response.ok) {
        throw new Error(`Erreur réseau HTTP Apps Script : ${response.statusText}`);
    }

    return await response.json();
}

// Utility Validation Email
function isValidEmail(email) {
    return typeof email === 'string' && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
}


// --- ENDPOINTS API UNIFIÉS ---

// 1. WORKFLOW EMAIL (Conserve Apps Script pour l'envoi Gmail)
app.post('/api/send-email', async (req, res) => {
    const { email } = req.body;

    if (!isValidEmail(email)) {
        return res.status(400).json({ 
            success: false, 
            message: "Format d'adresse e-mail invalide." 
        });
    }

    try {
        const result = await callAppsScript('sendMail', { email: email.trim() });
        res.status(result.success ? 200 : 400).json(result);
    } catch (error) {
        console.error("Erreur API Send-Email:", error);
        res.status(500).json({ success: false, message: error.message });
    }
});


// 2. ASSISTANT SAV (Priorité FastAPI/Scikit-Learn -> Fallback Apps Script)
app.post('/api/ia-sav', async (req, res) => {
    const { message } = req.body;

    if (!message || typeof message !== 'string' || !message.trim()) {
        return res.status(400).json({ success: false, message: "Le message SAV ne peut pas être vide." });
    }

    const cleanMsg = message.trim();

    try {
        // Étape 1 : Interrogation du modèle ML Python
        const mlResult = await callFastAPI(cleanMsg);

        if (mlResult && mlResult.success && mlResult.confidence >= 0.35) {
            return res.json({
                success: true,
                message: "Analyse effectuée via le moteur ML Scikit-Learn.",
                confidence: mlResult.confidence,
                intent: mlResult.intent,
                data: { reply: mlResult.reply }
            });
        }

        // Étape 2 : Fallback vers Google Apps Script
        const gasResult = await callAppsScript('iaSAV', { message: cleanMsg });
        res.status(gasResult.success ? 200 : 400).json(gasResult);

    } catch (error) {
        console.error("Erreur API IA-SAV:", error);
        res.status(500).json({ 
            success: false, 
            message: "Service SAV indisponible.", 
            data: { reply: "Une erreur est survenue lors de l'analyse de votre demande." } 
        });
    }
});


// 3. JUMEAU NUMÉRIQUE (Priorité FastAPI/Scikit-Learn -> Fallback Apps Script)
app.post('/api/ia-twin', async (req, res) => {
    const { question } = req.body;

    if (!question || typeof question !== 'string' || !question.trim()) {
        return res.status(400).json({ success: false, message: "La question ne peut pas être vide." });
    }

    const cleanQuestion = question.trim();

    try {
        // Étape 1 : Interrogation du modèle ML Python
        const mlResult = await callFastAPI(cleanQuestion);

        if (mlResult && mlResult.success && mlResult.confidence >= 0.35) {
            return res.json({
                success: true,
                message: "Réponse générée par l'IA (Scikit-Learn).",
                confidence: mlResult.confidence,
                intent: mlResult.intent,
                data: { reply: mlResult.reply }
            });
        }

        // Étape 2 : Fallback vers Google Apps Script
        const gasResult = await callAppsScript('iaTwin', { question: cleanQuestion });
        res.status(gasResult.success ? 200 : 400).json(gasResult);

    } catch (error) {
        console.error("Erreur API IA-Twin:", error);
        res.status(500).json({ 
            success: false, 
            message: "Jumeau Numérique hors-ligne.", 
            data: { reply: "Désolé, problème de connexion au serveur de réponses." } 
        });
    }
});


// CAPTURE DE TOUTES LES AUTRES ROUTES (Single Page Application)
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// DÉMARRAGE DU SERVEUR
app.listen(PORT, '0.0.0.0', () => {
    console.log(`✅ Serveur Node.js démarré sur le port ${PORT}`);
    console.log(`🔗 Endpoint FastAPI configuré sur : ${FASTAPI_URL}`);
});
