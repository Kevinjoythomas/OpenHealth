module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    res.status(405).json({ ok: false, error: 'Method not allowed' });
    return;
  }

  let body = req.body;
  if (typeof body === 'string') {
    try { body = JSON.parse(body); } catch (e) {
      res.status(400).json({ ok: false, error: 'Invalid JSON' });
      return;
    }
  }

  const rater = body && body.rater ? body.rater : {};
  const ratings = body && body.ratings ? body.ratings : [];
  const completed = body && typeof body.completed === 'number' ? body.completed : ratings.filter(r => r.rating).length;
  const total = body && typeof body.total === 'number' ? body.total : ratings.length;

  const clean = (s) => (s || '').replace(/^﻿/, '').trim();
  const RESEND_API_KEY = clean(process.env.RESEND_API_KEY);
  const NOTIFY_TO = clean(process.env.NOTIFY_TO);
  const FROM_ADDRESS = clean(process.env.RESEND_FROM) || 'onboarding@resend.dev';

  if (!RESEND_API_KEY || !NOTIFY_TO) {
    res.status(500).json({ ok: false, error: 'Email not configured on server' });
    return;
  }

  const stamp = new Date().toUTCString();
  const subject = `OpenHealth clinician review submitted — ${rater.name || 'unnamed'} (${completed}/${total})`;
  const summaryRows = [
    ['Name', rater.name || '(blank)'],
    ['Role', rater.role || '(blank)'],
    ['Years', rater.years || '(blank)'],
    ['Area', rater.area || '(blank)'],
    ['Workplace', rater.workplace || '(blank)'],
    ['Completed', `${completed}/${total}`],
    ['Received', stamp],
  ];
  const summaryHtml = `<table>${summaryRows.map(([k, v]) => `<tr><td style="padding:2px 10px 2px 0;color:#555;">${k}</td><td><b>${v}</b></td></tr>`).join('')}</table>`;
  const jsonPayload = JSON.stringify({ rater, ratings, completed, total, received_at: stamp }, null, 1);
  const base64Attachment = Buffer.from(jsonPayload, 'utf-8').toString('base64');
  const safeName = (rater.name || 'unnamed').replace(/[^a-z0-9]/gi, '_');

  try {
    const resp = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${RESEND_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: FROM_ADDRESS,
        to: [NOTIFY_TO],
        subject,
        html: `<h2>OpenHealth clinician review submitted</h2>${summaryHtml}<p>Full ratings JSON attached.</p>`,
        attachments: [
          {
            filename: `human_ratings_${safeName}.json`,
            content: base64Attachment,
          },
        ],
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => '');
      res.status(500).json({ ok: false, error: 'Failed to send email', detail: errText });
      return;
    }

    res.status(200).json({ ok: true });
  } catch (err) {
    res.status(500).json({ ok: false, error: 'Failed to send email', detail: String(err) });
  }
};
