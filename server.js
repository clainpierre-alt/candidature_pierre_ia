const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
// Render fournit automatiquement la variable PORT
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

const APPS_SCRIPT_URL = process.env.APPS_SCRIPT_URL;

// Vérification au démarrage (sans faire crasher le serveur)
if (!APPS_SCRIPT_URL) {
    console.warn("⚠️ ATTENTION : La variable APPS_SCRIPT_URL n'est pas définie dans l'onglet Environment de Render.");
}

// Helper pour contacter Google Apps Script
async function callAppsScript(action, payload) {
    if (!APPS_SCRIPT_URL) {
        throw new Error("URL Apps Script manquante dans la configuration.");
    }
    const response = await fetch(APPS_SCRIPT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...payload }),
        redirect: 'follow'
    });
    return await response.json();
}

// Route de vérification simple (Healthcheck)
app.get('/health', (req, res) => {
    res.status(200).send("OK");
});

// ENDPOINTS API
app.post('/api/send-email', async (req, res) => {
    try {
        const result = await callAppsScript('send-email', { email: req.body.email });
        res.status(200).json(result);
    } catch (error) {
        res.status(500).json({ error: error.message || "Erreur d'envoi de mail." });
    }
});

app.post('/api/ia-sav', async (req, res) => {
    try {
        const result = await callAppsScript('ia-sav', { message: req.body.message });
        res.status(200).json(result);
    } catch (error) {
        res.status(500).json({ reply: "Service SAV indisponible." });
    }
});

app.post('/api/ia-twin', async (req, res) => {
    try {
        const result = await callAppsScript('ia-twin', { question: req.body.question });
        res.status(200).json(result);
    } catch (error) {
        res.status(500).json({ reply: "Jumeau numérique hors-ligne." });
    }
});

// Lancement de l'écoute sur le port dynamique
app.listen(PORT, '0.0.0.0', () => {
    console.log(`✅ Serveur prêt et à l'écoute sur le port ${PORT}`);
});
