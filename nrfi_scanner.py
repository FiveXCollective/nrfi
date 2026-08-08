#!/usr/bin/env python3
"""
NRFI Edge Scanner — v1
----------------------
Run-once daily script (Railway native cron friendly).

What it does:
  1. Pulls today's MLB slate + probable starters (MLB Stats API, free, no key).
  2. For each starter, computes their 1st-inning "scoreless half" rate from
     Statcast pitch-level data (via pybaseball), shrunk toward a league prior.
  3. Models each game's P(NRFI) = P(home half scoreless) * P(away half scoreless).
  4. Ranks games, flags leans, and suggests 2-3 leg parlays.
  5. Optionally loads a book line (NRFI/YRFI) to compute de-vigged edge + EV.
  6. Emails an HTML digest.

The model is deliberately simple and transparent. Park factors, weather, umpire,
opposing top-of-order quality, and confirmed lineups are documented v2 upgrades.

Edge lives in the MODEL beating the market price. Stacking legs into a parlay is
only +EV if EACH leg is +EV vs the book — the vig compounds per leg.
"""

import os
import sys
import json
import math
import smtplib
import datetime as dt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests

# pybaseball is heavy; import lazily and fail loud if missing.
try:
    from pybaseball import statcast_pitcher
    from pybaseball import cache as pyb_cache
    pyb_cache.enable()  # local caching within a run / between runs on warm FS
except Exception as e:  # pragma: no cover
    statcast_pitcher = None
    _PYB_ERR = e

# ----------------------------------------------------------------------------
# Config (all overridable via env vars)
# ----------------------------------------------------------------------------
MLB_API = "https://statsapi.mlb.com/api/v1"

TODAY = os.getenv("SCAN_DATE") or dt.date.today().isoformat()
SEASON_START = os.getenv("SEASON_START", f"{TODAY[:4]}-03-01")

# League baseline: a single half-inning is scoreless ~72% of the time
# (full-inning NRFI ~49-52% => per-half ~sqrt of that).
LEAGUE_HALF_SCORELESS = float(os.getenv("LEAGUE_HALF_SCORELESS", "0.72"))
PRIOR_K = float(os.getenv("PRIOR_K", "10"))          # shrinkage strength (pseudo-starts)
# League rate that a team is held scoreless in the 1st inning (mirror of the ~28%
# of games a team scores in the 1st). Used for the top-of-order adjustment.
LEAGUE_OFF_SCORELESS = float(os.getenv("LEAGUE_OFF_SCORELESS", "0.72"))
TEAM_PRIOR_K = float(os.getenv("TEAM_PRIOR_K", "8"))  # shrinkage for team offense (pseudo-games)
MIN_STARTS_CONF = int(os.getenv("MIN_STARTS_CONF", "5"))  # both starters need this for HIGH conf
MODEL_MIN_PROB = float(os.getenv("MODEL_MIN_PROB", "0.55"))  # threshold to call it a "lean"
EDGE_MIN = float(os.getenv("EDGE_MIN", "0.03"))      # +3pp edge vs book to call it a play

# Optional: JSON file of book lines to compute edge. Format:
# [{"match": "Away Team @ Home Team", "nrfi": -115, "yrfi": -105}, ...]
ODDS_FILE = os.getenv("ODDS_FILE", "")

# Email delivery. Two transports, tried in this order:
#   1. Resend HTTPS API (RESEND_API_KEY) — works on hosts that block SMTP
#      egress (e.g. Railway blocks ports 25/465/587). Uses port 443.
#   2. SMTP (SMTP_USER + SMTP_PASS) — fine locally or on hosts that allow it.
# If neither is configured, the digest is printed to stdout.
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")

# SMTP (fallback) — same pattern as the ATR scanner
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_TO = os.getenv("EMAIL_TO", "jxenakis@hotmail.com")
# From address. For Resend this must be a verified-domain sender, or
# "onboarding@resend.dev" (which can only deliver to your own account email).
EMAIL_FROM = os.getenv("EMAIL_FROM") or SMTP_USER or "onboarding@resend.dev"


# ----------------------------------------------------------------------------
# Data: today's slate + probable pitchers
# ----------------------------------------------------------------------------
def fetch_slate(date_str):
    r = requests.get(
        f"{MLB_API}/schedule",
        params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team"},
        timeout=30,
    )
    r.raise_for_status()
    games = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            home, away = g["teams"]["home"], g["teams"]["away"]
            hp = home.get("probablePitcher") or {}
            ap = away.get("probablePitcher") or {}
            games.append({
                "game_pk": g["gamePk"],
                "start": g.get("gameDate"),
                "home_team": home["team"]["name"],
                "away_team": away["team"]["name"],
                "home_id": home["team"]["id"],
                "away_id": away["team"]["id"],
                "home_pitcher": hp.get("fullName"),
                "home_pid": hp.get("id"),
                "away_pitcher": ap.get("fullName"),
                "away_pid": ap.get("id"),
            })
    return games


# ----------------------------------------------------------------------------
# Model: per-pitcher 1st-inning scoreless rate from Statcast
# ----------------------------------------------------------------------------
def pitcher_first_inning(pid, end_str):
    """Return {'starts', 'clean'} for the pitcher's 1st inning, or None."""
    if not pid or statcast_pitcher is None:
        return None
    try:
        df = statcast_pitcher(SEASON_START, end_str, pid)
    except Exception:
        return None
    if df is None or df.empty or "inning" not in df.columns:
        return None
    d1 = df[df["inning"] == 1].copy()
    if d1.empty:
        return None
    # A run is allowed when the batting team's score increases during the pitch.
    d1["runs"] = (d1["post_bat_score"] - d1["bat_score"]).clip(lower=0)
    by_game = d1.groupby("game_pk")["runs"].sum()
    starts = int(by_game.shape[0])
    clean = int((by_game == 0).sum())
    return {"starts": starts, "clean": clean}


def shrink(clean, starts):
    """Beta-style shrink toward the league half-inning scoreless prior."""
    return (clean + PRIOR_K * LEAGUE_HALF_SCORELESS) / (starts + PRIOR_K)


_TEAM_CACHE = {}

def team_offense_scoreless(team_id, end_str):
    """How often this team is held scoreless in the 1st inning, season-to-date.
    One call per team (memoized). Falls back to the league prior on any failure."""
    if team_id in _TEAM_CACHE:
        return _TEAM_CACHE[team_id]
    rate = LEAGUE_OFF_SCORELESS  # graceful fallback
    try:
        r = requests.get(
            f"{MLB_API}/schedule",
            params={
                "sportId": 1, "teamId": team_id, "gameType": "R",
                "startDate": SEASON_START, "endDate": end_str, "hydrate": "linescore",
            },
            timeout=30,
        )
        r.raise_for_status()
        scored = total = 0
        for d in r.json().get("dates", []):
            for g in d.get("games", []):
                if (g.get("status", {}).get("abstractGameState") != "Final"):
                    continue
                innings = (g.get("linescore") or {}).get("innings") or []
                if not innings:
                    continue
                first = innings[0]
                is_home = g["teams"]["home"]["team"]["id"] == team_id
                side = first.get("home" if is_home else "away") or {}
                runs = side.get("runs")
                if runs is None:
                    continue
                total += 1
                if runs > 0:
                    scored += 1
        if total > 0:
            # shrink the SCORE rate toward league (0.28), then convert to scoreless
            score_rate = (scored + TEAM_PRIOR_K * (1 - LEAGUE_OFF_SCORELESS)) / (total + TEAM_PRIOR_K)
            rate = 1 - score_rate
    except Exception:
        pass
    _TEAM_CACHE[team_id] = rate
    return rate


def log5(p_pitcher, p_offense, p_league=LEAGUE_HALF_SCORELESS):
    """Combine a pitcher's scoreless-half rate with the opposing offense's
    scoreless rate, anchored to league average (Bill James log5). Returns the
    probability the half-inning stays scoreless."""
    pa = min(max(p_pitcher, 0.01), 0.99)
    pb = min(max(p_offense, 0.01), 0.99)
    pl = min(max(p_league, 0.02), 0.98)
    num = pa * pb / pl
    den = num + (1 - pa) * (1 - pb) / (1 - pl)
    return num / den if den else pa


def fair_american(p):
    p = min(max(p, 0.01), 0.99)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def american_to_decimal(a):
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


def devig_two_way(nrfi_a, yrfi_a):
    """Fair NRFI probability implied by a two-sided market (vig removed)."""
    inr, iyr = 1 / american_to_decimal(nrfi_a), 1 / american_to_decimal(yrfi_a)
    return inr / (inr + iyr)


def ev_per_unit(p, american):
    d = american_to_decimal(american)
    return p * (d - 1) - (1 - p)


# ----------------------------------------------------------------------------
# Build the board
# ----------------------------------------------------------------------------
def load_odds():
    if not ODDS_FILE or not os.path.exists(ODDS_FILE):
        return {}
    try:
        rows = json.load(open(ODDS_FILE))
        return {r["match"]: r for r in rows}
    except Exception:
        return {}


def build_board(games, odds):
    board = []
    for g in games:
        if not g["home_pid"] or not g["away_pid"]:
            continue  # probables not posted yet
        h = pitcher_first_inning(g["home_pid"], TODAY)
        a = pitcher_first_inning(g["away_pid"], TODAY)
        if h is None or a is None:
            continue

        # Base pitcher scoreless-half rates
        ph = shrink(h["clean"], h["starts"])   # home starter
        pa = shrink(a["clean"], a["starts"])   # away starter

        # Opposing offense: home starter pitches TOP 1st vs the AWAY bats;
        # away starter pitches BOTTOM 1st vs the HOME bats.
        away_off = team_offense_scoreless(g["away_id"], TODAY)
        home_off = team_offense_scoreless(g["home_id"], TODAY)

        p_top = log5(ph, away_off)   # top of 1st stays scoreless
        p_bot = log5(pa, home_off)   # bottom of 1st stays scoreless
        p_nrfi = p_top * p_bot       # both halves scoreless

        min_starts = min(h["starts"], a["starts"])
        conf = "HIGH" if min_starts >= MIN_STARTS_CONF else ("MED" if min_starts >= 2 else "LOW")

        row = {
            "match": f'{g["away_team"]} @ {g["home_team"]}',
            "home_p": g["home_pitcher"], "away_p": g["away_pitcher"],
            "home_line": f'{h["clean"]}/{h["starts"]} clean · opp off {1-away_off:.0%} score',
            "away_line": f'{a["clean"]}/{a["starts"]} clean · opp off {1-home_off:.0%} score',
            "p_nrfi": p_nrfi,
            "fair": fair_american(p_nrfi),
            "conf": conf,
            "min_starts": min_starts,
            "edge": None, "book": None, "ev": None,
        }

        book = odds.get(row["match"])
        if book and "nrfi" in book and "yrfi" in book:
            fair_mkt = devig_two_way(book["nrfi"], book["yrfi"])
            row["book"] = book["nrfi"]
            row["edge"] = p_nrfi - fair_mkt          # model prob minus de-vigged market prob
            row["ev"] = ev_per_unit(p_nrfi, book["nrfi"])
        board.append(row)

    board.sort(key=lambda r: r["p_nrfi"], reverse=True)
    return board


def suggest_parlays(board):
    """Legs: leaning NRFI, with at least MED confidence. If a book line is
    present, require +EV. Otherwise rank by model prob."""
    legs = [
        r for r in board
        if r["p_nrfi"] >= MODEL_MIN_PROB and r["conf"] != "LOW"
        and (r["edge"] is None or r["edge"] >= EDGE_MIN)
    ]
    out = {}
    for n in (2, 3, 4):
        if len(legs) >= n:
            picks = legs[:n]
            p = math.prod(x["p_nrfi"] for x in picks)
            out[n] = {"legs": picks, "p": p, "fair": fair_american(p)}
    return out


# ----------------------------------------------------------------------------
# Render + send
# ----------------------------------------------------------------------------
def render_html(board, parlays):
    def pct(x):
        return f"{x*100:.1f}%"

    def edge_cell(r):
        if r["edge"] is None:
            return '<span style="color:#7a8699">—</span>'
        col = "#2ecc71" if r["edge"] >= EDGE_MIN else ("#e74c3c" if r["edge"] < 0 else "#f1c40f")
        return f'<span style="color:{col};font-weight:600">{r["edge"]*100:+.1f}pp</span>'

    rows = ""
    for r in board:
        rows += f"""
        <tr>
          <td style="padding:10px 8px;border-bottom:1px solid #1f2733">{r['match']}</td>
          <td style="padding:10px 8px;border-bottom:1px solid #1f2733;font-size:12px;color:#9fb0c3">
            {r['away_p']} <span style="color:#5c6b7d">({r['away_line']})</span><br>
            {r['home_p']} <span style="color:#5c6b7d">({r['home_line']})</span>
          </td>
          <td align="center" style="padding:10px 8px;border-bottom:1px solid #1f2733;font-weight:700">{pct(r['p_nrfi'])}</td>
          <td align="center" style="padding:10px 8px;border-bottom:1px solid #1f2733">{r['fair']:+d}</td>
          <td align="center" style="padding:10px 8px;border-bottom:1px solid #1f2733">{r['book'] if r['book'] is not None else '—'}</td>
          <td align="center" style="padding:10px 8px;border-bottom:1px solid #1f2733">{edge_cell(r)}</td>
          <td align="center" style="padding:10px 8px;border-bottom:1px solid #1f2733;font-size:11px;color:#9fb0c3">{r['conf']}</td>
        </tr>"""

    parlay_html = ""
    for n, p in sorted(parlays.items()):
        leg_lines = "".join(
            f'<li style="margin:2px 0">{x["match"]} — <b>{pct(x["p_nrfi"])}</b> NRFI</li>'
            for x in p["legs"]
        )
        parlay_html += f"""
        <div style="background:#12181f;border:1px solid #1f2733;border-radius:10px;padding:14px 16px;margin:10px 0">
          <div style="font-weight:700;color:#4aa3ff;margin-bottom:6px">{n}-Leg Parlay
            &nbsp;·&nbsp; Model hit {pct(p['p'])} &nbsp;·&nbsp; Fair {p['fair']:+d}</div>
          <ul style="margin:6px 0 0 18px;padding:0;color:#c9d6e3;font-size:13px">{leg_lines}</ul>
        </div>"""
    if not parlay_html:
        parlay_html = '<div style="color:#7a8699">No legs cleared the threshold today.</div>'

    has_odds = any(r["book"] is not None for r in board)
    odds_note = "" if has_odds else (
        '<div style="background:#2a1f12;border:1px solid #5c4326;border-radius:8px;'
        'padding:10px 14px;color:#e0b877;font-size:12px;margin:0 0 16px">'
        "No book lines loaded — showing model fair odds only. Wire an NRFI-capable odds "
        "source (or drop a lines JSON via ODDS_FILE) to get de-vigged edge + EV.</div>"
    )

    return f"""\
<!doctype html><html><body style="margin:0;background:#0b0f14;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
<div style="max-width:720px;margin:0 auto;padding:24px 16px;color:#e6edf3">
  <div style="font-size:22px;font-weight:800;letter-spacing:.3px">⚾ NRFI Edge Digest</div>
  <div style="color:#7a8699;font-size:13px;margin:2px 0 18px">{TODAY} · {len(board)} games modeled</div>
  {odds_note}
  <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:13px">
    <thead><tr style="color:#7a8699;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px">
      <th style="padding:6px 8px">Game</th><th style="padding:6px 8px">Starters (1st-inn history)</th>
      <th style="padding:6px 8px" align="center">Model NRFI</th><th style="padding:6px 8px" align="center">Fair</th>
      <th style="padding:6px 8px" align="center">Book</th><th style="padding:6px 8px" align="center">Edge</th>
      <th style="padding:6px 8px" align="center">Conf</th>
    </tr></thead><tbody>{rows}</tbody>
  </table>

  <div style="font-size:16px;font-weight:800;margin:26px 0 4px">Suggested Parlays</div>
  <div style="color:#7a8699;font-size:12px;margin-bottom:6px">
    Only +EV if each leg beats its book price — the vig compounds per leg. Confirm lineups before firing.</div>
  {parlay_html}

  <div style="color:#5c6b7d;font-size:11px;margin-top:24px;line-height:1.5">
    Model: each 1st-inning half = starter's scoreless rate (Statcast, season-to-date, shrunk to a
    {LEAGUE_HALF_SCORELESS:.0%} prior) blended via log5 with the opposing offense's 1st-inning scoring
    rate. P(NRFI) = top-half scoreless × bottom-half scoreless.
    v2 TODO: park, weather, umpire, confirmed lineups, half-correlation.
    Not financial advice — bet within your means.
  </div>
</div></body></html>"""


def _send_via_resend(subject, html):
    """Send through the Resend HTTPS API (port 443). Raises on any error."""
    r = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"from": EMAIL_FROM, "to": [EMAIL_TO], "subject": subject, "html": html},
        timeout=30,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Resend API error {r.status_code}: {r.text}")
    print(f"Sent digest to {EMAIL_TO} via Resend (id={r.json().get('id')})")


def _send_via_smtp(subject, html):
    """Send through SMTP. Note: many hosts (Railway) block SMTP egress."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
    print(f"Sent digest to {EMAIL_TO} via SMTP")


def send_email(html):
    subject = f"⚾ NRFI Edge Digest — {TODAY}"
    if RESEND_API_KEY:
        _send_via_resend(subject, html)
        return
    if SMTP_USER and SMTP_PASS:
        _send_via_smtp(subject, html)
        return
    print("No email transport configured (set RESEND_API_KEY or SMTP creds) —"
          " printing digest to stdout instead.\n")
    print(html[:2000])


def main():
    if statcast_pitcher is None:
        print(f"pybaseball not available: {_PYB_ERR}", file=sys.stderr)
        sys.exit(1)
    print(f"Scanning {TODAY} (season start {SEASON_START})...")
    games = fetch_slate(TODAY)
    print(f"  {len(games)} games on the slate")
    if not games:
        send_email(f"<p>No games scheduled for {TODAY}.</p>")
        return
    odds = load_odds()
    board = build_board(games, odds)
    print(f"  {len(board)} games modeled")
    parlays = suggest_parlays(board)
    html = render_html(board, parlays)
    send_email(html)


if __name__ == "__main__":
    main()
