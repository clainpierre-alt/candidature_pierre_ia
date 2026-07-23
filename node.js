const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const APPS_SCRIPT_URL = process.env.APPS_SCRIPT_URL;

// Helper de communication centralisé avec Apps Script
async function callAppsScript(action, payload) {
    if (!APPS_SCRIPT_URL) {
        throw new Error("Variable APPS_SCRIPT_URL non configurée sur le serveur Render.");
    }
    
    const response = await fetch(APPS_SCRIPT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...payload }),
        redirect: 'follow'
    });

    if (!response.ok) {
        throw new Error(`Erreur réseau HTTP : ${response.statusText}`);
    }

    return await response.json();
}

// Utility Validation Email
function isValidEmail(email) {
    return typeof email === 'string' && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
}

// --- ENDPOINTS API UNIFIÉS ---

// 1. Workflow Email
app.post('/api/send-email', async (req, res) => {
    const { email } = req.body;

    if (!isValidEmail(email)) {
        return res.status(400).json({ 
            success: false, 
            message: "Format d'adresse email invalide." 
        });
    }

    try {
        const result = await callAppsScript('send-email', { email: email.trim() });
        res.status(result.success ? 200 : 400).json(result);
    } catch (error) {
        console.error("Erreur API Send-Email:", error);
        res.status(500).json({ success: false, message: error.message });
    }
});

// 2. Assistant SAV
app.post('/api/ia-sav', async (req, res) => {
    const { message } = req.body;

    if (!message || typeof message !== 'string' || !message.trim()) {
        return res.status(400).json({ success: false, message: "Le message SAV ne peut pas être vide." });
    }

    try {
        const result = await callAppsScript('ia-sav', { message: message.trim() });
        res.status(result.success ? 200 : 400).json(result);
    } catch (error) {
        console.error("Erreur API IA-SAV:", error);
        res.status(500).json({ success: false, message: "Service SAV indisponible.", data: { reply: "Erreur serveur." } });
    }
});

// 3. Jumeau Numérique
app.post('/api/ia-twin', async (req, res) => {
    const { question } = req.body;

    if (!question || typeof question !== 'string' || !question.trim()) {
        return res.status(400).json({ success: false, message: "La question ne peut pas être vide." });
    }

    try {
        const result = await callAppsScript('ia-twin', { question: question.trim() });
        res.status(result.success ? 200 : 400).json(result);
    } catch (error) {
        console.error("Erreur API IA-Twin:", error);
        res.status(500).json({ success: false, message: "Jumeau Numérique hors-ligne.", data: { reply: "Désolé, problème de connexion au serveur." } });
    }
});

// Capture de toutes les autres routes
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`✅ Serveur prêt sur le port ${PORT}`);
});
