const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

const APPS_SCRIPT_URL = process.env.APPS_SCRIPT_URL;

// Helper pour router vers Google Apps Script
async function callAppsScript(action, payload) {
    const response = await fetch(APPS_SCRIPT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...payload }),
        redirect: 'follow' // Indispensable pour suivre les redirections de Google
    });
    return await response.json();
}

// ENDPOINTS API
app.post('/api/send-email', async (req, res) => {
    try {
        const result = await callAppsScript('send-email', { email: req.body.email });
        res.status(200).json(result);
    } catch (error) {
        res.status(500).json({ error: "Erreur d'envoi de mail via Apps Script" });
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

app.listen(PORT, () => console.log(`Serveur actif sur le port ${PORT}`));
