const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const app = express();

// Render fournit automatiquement la variable PORT
const PORT = process.env.PORT || 3000;

// Middlewares
app.use(cors());
app.use(express.json());

// Servir les fichiers statiques du dossier public
app.use(express.static(path.join(__dirname, 'public')));

const APPS_SCRIPT_URL = process.env.APPS_SCRIPT_URL;

// Vérification au démarrage dans les logs Render
if (!APPS_SCRIPT_URL) {
    console.warn("⚠️ ATTENTION : La variable APPS_SCRIPT_URL n'est pas encore définie dans Render.");
}

// Helper pour router les requêtes vers Google Apps Script
async function callAppsScript(action, payload) {
    if (!APPS_SCRIPT_URL) {
        throw new Error("URL Google Apps Script manquante dans les variables d'environnement.");
    }
    const response = await fetch(APPS_SCRIPT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...payload }),
        redirect: 'follow' // Suivre la redirection automatique de Google Apps Script
    });
    return await response.json();
}

// Route de test rapide (Healthcheck)
app.get('/health', (req, res) => {
    res.status(200).send("OK");
});

// --- ENDPOINTS API ---

// 1. Workflow Email
app.post('/api/send-email', async (req, res) => {
    try {
        const result = await callAppsScript('send-email', { email: req.body.email });
        res.status(200).json(result);
    } catch (error) {
        console.error("Erreur Email:", error);
        res.status(500).json({ error: error.message || "Erreur lors de l'envoi du mail via Apps Script." });
    }
});

// 2. Assistant SAV
app.post('/api/ia-sav', async (req, res) => {
    try {
        const result = await callAppsScript('ia-sav', { message: req.body.message });
        res.status(200).json(result);
    } catch (error) {
        console.error("Erreur SAV:", error);
        res.status(500).json({ reply: "Service SAV indisponible." });
    }
});

// 3. Jumeau Numérique
app.post('/api/ia-twin', async (req, res) => {
    try {
        const result = await callAppsScript('ia-twin', { question: req.body.question });
        res.status(200).json(result);
    } catch (error) {
        console.error("Erreur Twin:", error);
        res.status(500).json({ reply: "Jumeau numérique momentanément hors-ligne." });
    }
});

// Redirection par défaut sur l'index.html
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Lancement du serveur
app.listen(PORT, '0.0.0.0', () => {
    console.log(`✅ Serveur prêt et à l'écoute sur le port ${PORT}`);
});
