#!/usr/bin/env node
/**
 * WhatsApp pairing code generator.
 * Generates a code the user enters on their phone instead of scanning QR.
 */

const { makeWASocket, useMultiFileAuthState, fetchLatestBaileysVersion, DisconnectReason } = require('@whiskeysockets/baileys');
const pino = require('pino');
const path = require('path');
const { Boom } = require('@hapi/boom');

const SESSION_DIR = process.env.SESSION_DIR || path.join(process.env.HOME, '.hermes', 'whatsapp', 'session');
const PHONE_NUMBER = process.env.WHATSAPP_PHONE || '56975296398';

const logger = pino({ level: 'error' });

async function start() {
  const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);
  const { version } = await fetchLatestBaileysVersion();

  const sock = makeWASocket({
    version,
    auth: state,
    logger,
    printQRInTerminal: false,
    browser: ['Hermes Agent', 'Chrome', '120.0'],
    syncFullHistory: false,
    markOnlineOnConnect: false,
    getMessage: async (key) => ({ conversation: '' }),
  });

  sock.ev.on('creds.update', saveCreds);

  let pairingRequested = false;

  sock.ev.on('connection.update', async (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr && !pairingRequested) {
      pairingRequested = true;
      try {
        const code = await sock.requestPairingCode(PHONE_NUMBER);
        // The code comes in groups of 2 digits separated by dashes
        const displayCode = String(code).match(/.{1,4}/g)?.join('-') || code;
        
        console.log('');
        console.log('╔══════════════════════════════════════════════╗');
        console.log('║                                              ║');
        console.log('║   📱 WHATSAPP PAIRING CODE                   ║');
        console.log('║                                              ║');
        console.log(`║   🔑  ${displayCode.padEnd(30)}  ║`);
        console.log('║                                              ║');
        console.log('╚══════════════════════════════════════════════╝');
        console.log('');
        console.log('📌 Steps:');
        console.log('1. Open WhatsApp on your phone (' + PHONE_NUMBER + ')');
        console.log('2. Go to Settings → Linked Devices → Link a Device');
        console.log('3. Choose "Link with phone number" (if available)');
        console.log('4. Enter the code above');
        console.log('');
        console.log('⏳ Waiting for phone to pair...');
        console.log('');
      } catch (err) {
        console.error('❌ Pairing code request failed:', err.message);
        console.log('   Falling back to QR code scan...');
      }
    }

    if (connection === 'open') {
      saveCreds();
      console.log('');
      console.log('✅ WhatsApp paired and connected successfully!');
      console.log('📁 Session saved to: ' + SESSION_DIR);
      process.exit(0);
    }

    if (connection === 'close') {
      const boom = lastDisconnect?.error ? new Boom(lastDisconnect.error) : null;
      const reason = boom?.output?.statusCode;
      if (reason === DisconnectReason.loggedOut) {
        console.log('❌ Logged out. Delete session and try again.');
        process.exit(1);
      }
    }
  });
}

start().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});