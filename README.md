# NRFI Edge Scanner

Daily MLB "No Run First Inning" model. Pulls today's slate + probable starters,
scores each game's P(NRFI) from real 1st-inning data, ranks them, suggests
parlays, and emails an HTML digest. Run-once script, built for Railway cron.

## The one thing to internalize
The edge is the **model beating the book price**, not "safe favorites." Stacking
low-odds legs into a parlay pays the vig once per leg — four true-90% legs only
cash ~66% of the time. So the scanner ranks by model probability and, when you
feed it a book line, by **de-vigged edge + EV**. Only bet legs that beat their price.

## Local run
```bash
pip install -r requirements.txt
cp .env.example .env          # fill in SMTP creds
export $(grep -v '^#' .env | xargs)
python nrfi_scanner.py
```
No SMTP creds? It prints the digest to stdout so you can eyeball it.
First run pulls Statcast for ~30 starters (~1-2 min); pybaseball caches after.

## Deploy on Railway (GitHub)
1. Push this folder to a GitHub repo.
2. Railway → New Project → Deploy from GitHub → pick the repo.
   Nixpacks auto-detects Python and installs `requirements.txt`.
3. Service → **Variables** → add everything from `.env.example`.
   **Important:** Railway blocks outbound SMTP (ports 25/465/587), so Gmail/SMTP
   sending fails there. Use **Resend** instead — set `RESEND_API_KEY` (sends over
   HTTPS/443). The script auto-prefers Resend when that key is present and only
   falls back to SMTP otherwise. Set `EMAIL_FROM` to a Resend verified-domain
   sender, or `onboarding@resend.dev` (which only delivers to your account email).
4. Service → **Settings → Cron Schedule** → enter a 5-field crontab in **UTC**.
   - `0 15 * * *` = **11:00 AM ET (EDT)** — lineups are firming up by then, which
     the top-of-order adjustment relies on. This is the recommended slot.
   - Note: EDT is UTC-4. In standard time (Nov–Mar, no games anyway) it'd be UTC-5.
   - Min frequency is every 5 min; Railway starts the service, runs it, exits.
5. That's it. `restartPolicyType: NEVER` keeps it from relooping after it finishes.

## Real edge (odds)
The Odds API has no market literally named NRFI, but **1st-inning totals**
(`totals_1st_1_innings`) at a 0.5 line *are* NRFI: **Under 0.5 = NRFI**, Over 0.5 = YRFI.

Set `ODDS_API_KEY` and the scanner pulls them automatically:
- These are "additional markets", so they come from the per-event endpoint —
  ~1 credit per game (+1 to list events); a 15-game slate ≈ 16 credits/run.
  The free tier (500/mo) covers a once-daily run comfortably.
- For each game it takes the **best US price** (for EV) and the **median per-book
  de-vigged line** (the fair prob for edge). Digest columns: **Book** = best price +
  source, **Edge** = model − fair, **EV** = per-unit return at the best price.
- Coverage varies by book/game; unpriced games fall back to model-only.

**Read the edge honestly.** A model showing +15–35pp "edges" is almost always the
*model* being mis-calibrated, not free money — real NRFI edges are a few points.
Treat large disagreements with the de-vigged market as a flag to distrust the model,
and backtest calibration before betting to price. Not financial advice.

Fallback without the API — drop a `lines.json` and set `ODDS_FILE=lines.json`:
```json
[{"match": "New York Yankees @ Boston Red Sox", "nrfi": -120, "yrfi": +100}]
```
`match` must be exactly `"Away Team @ Home Team"` as MLB names them.

## Model (v1.1)
Each half-inning's scoreless probability blends two inputs via **log5**:
- the starter's 1st-inning scoreless rate (Statcast, season-to-date, shrunk to a 72% prior), and
- the opposing offense's 1st-inning scoring rate (from that team's game linescores).

Matchup is handled correctly: the home starter faces the **away** bats in the top
of the 1st, the away starter faces the **home** bats in the bottom.
`P(NRFI) = P(top scoreless) × P(bottom scoreless)`. The digest shows each
opponent's "opp off X% score" so you can see the adjustment working.

### Confirmed lineups (v2.1)
The digest shows a **Lineups** status per game — "✓ Set" once both batting orders
are posted (they firm up a few hours pre-game), else "Pending" — and lists the
opposing top-of-order (`vs …`) each starter faces. Set `REQUIRE_CONFIRMED_LINEUPS=1`
on a late-morning/pre-game re-run to keep pending-lineup games out of the suggested
parlays, so you don't stack legs on a stale slate (probables can scratch).

### v2 upgrades (in priority order)
1. ~~Opposing top-of-order quality~~ ✅ done (team-level; player-level is a further refinement).
2. ~~Confirmed lineups~~ ✅ done (status + parlay gating; see above).
3. Live NRFI odds feed → auto de-vigged edge + EV (biggest betting win).
4. Park factor + weather (wind/temp).
5. Player-level top-of-order (actual 1–3 hitters vs the starter's hand).
6. Home-plate umpire zone.
7. Half-inning correlation instead of strict independence.
8. Then clone the whole skeleton for K-props and SB markets.
