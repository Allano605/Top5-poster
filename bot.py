"""
Top 5 Match Pick — Auto Alert Bot (Web Service version for Render free tier)
-----------------------------------------------------------------------------
Same logic as before: pulls TODAY's fixtures + odds + form from API-Football
for the top 5 leagues, and sends a Telegram alert 30 minutes before kickoff.

WHAT CHANGED FROM main.py:
- Wrapped in a tiny Flask app so Render's free "Web Service" plan (no card
  required, 750 free hours/month) can run it, instead of needing a paid
  Cron Job or Background Worker.
- The fixture-checking loop now runs forever in a background thread,
  checking every 5 minutes on its own — same timing as before.
- Flask just serves one simple page at "/" so an uptime pinger (e.g.
  UptimeRobot, free, no card needed) can hit it every 5 min to stop
  Render from putting the service to sleep.

NOTHING about the alert logic, odds calculation, form calculation, or
Telegram message format has changed. This is purely a "keep it running
for free" wrapper.

SETUP ON RENDER:
1. This file replaces your old bot.py in the GitHub repo (Allano605/Top5-poster)
2. Environment Variables (already set, keep as is):
     BOT_TOKEN, CHANNEL_ID, FOOTBALL_API_KEY
3. Start Command on Render should be:
     gunicorn bot:app
   (or "python bot.py" also works, gunicorn is just more standard for Flask)
4. After deploy, go to uptimerobot.com (free signup, no card):
   - Add New Monitor -> HTTP(s)
   - URL: https://top5-poster.onrender.com
   - Monitoring interval: every 5 minutes
   This keeps Render from sleeping the service.
"""

import os
import json
import time
import threading
import requests
from datetime import datetime, timezone
from flask import Flask

# ---------- CONFIG ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
CHANNEL_ID = os.environ["CHANNEL_ID"]
API_KEY = os.environ["FOOTBALL_API_KEY"]

API_HOST = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}

SENT_FILE = "sent_matches.json"
ALERT_WINDOW_MIN = 25
ALERT_WINDOW_MAX = 35
CHECK_INTERVAL_SECONDS = 300  # 5 minutes


# ---------- SENT-TRACKING ----------
def load_sent():
    if os.path.exists(SENT_FILE):
        with open(SENT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_sent(sent):
    with open(SENT_FILE, "w") as f:
        json.dump(sent, f)

def cleanup_old_sent(sent):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {k: v for k, v in sent.items() if v.get("date") == today}


# ---------- API CALLS ----------
def get_current_season():
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1

def fetch_fixtures_today():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    season = get_current_season()
    all_fixtures = []
    for league_id, league_name in LEAGUES.items():
        resp = requests.get(
            f"{API_HOST}/fixtures",
            headers=HEADERS,
            params={"league": league_id, "season": season, "date": today},
        )
        data = resp.json().get("response", [])
        for fx in data:
            fx["_league_name"] = league_name
        all_fixtures.extend(data)
    return all_fixtures

def fetch_odds(fixture_id):
    resp = requests.get(
        f"{API_HOST}/odds",
        headers=HEADERS,
        params={"fixture": fixture_id},
    )
    data = resp.json().get("response", [])
    if not data:
        return None
    try:
        bookmaker = data[0]["bookmakers"][0]
        bets = bookmaker["bets"][0]["values"]
        odds = {v["value"]: float(v["odd"]) for v in bets}
        return {
            "home": odds.get("Home"),
            "draw": odds.get("Draw"),
            "away": odds.get("Away"),
        }
    except (KeyError, IndexError):
        return None

def fetch_last5_form(team_id):
    resp = requests.get(
        f"{API_HOST}/fixtures",
        headers=HEADERS,
        params={"team": team_id, "last": 5},
    )
    data = resp.json().get("response", [])
    form_str = ""
    goals_scored = []
    for fx in data:
        teams = fx["teams"]
        goals = fx["goals"]
        is_home = teams["home"]["id"] == team_id
        team_goals = goals["home"] if is_home else goals["away"]
        goals_scored.append(team_goals if team_goals is not None else 0)
        if teams["home"]["winner"] is None:
            form_str += "D"
        elif (is_home and teams["home"]["winner"]) or (not is_home and teams["away"]["winner"]):
            form_str += "W"
        else:
            form_str += "L"
    avg_goals = sum(goals_scored) / len(goals_scored) if goals_scored else 0
    return form_str or "N/A", avg_goals


# ---------- CALCULATIONS ----------
def implied_probabilities(odds):
    if not odds or not all([odds.get("home"), odds.get("draw"), odds.get("away")]):
        return None
    raw = {k: 1 / v for k, v in odds.items()}
    total = sum(raw.values())
    return {k: round((v / total) * 100) for k, v in raw.items()}

def build_combined_pick(home_avg_goals, away_avg_goals, home_name, away_name):
    total_avg = home_avg_goals + away_avg_goals
    if home_avg_goals >= 1 and away_avg_goals >= 1:
        return "Both Teams to Score – Yes"
    elif total_avg >= 2.8:
        return "Over 2.5 Goals"
    elif total_avg >= 1.8:
        return "Over 1.5 Goals"
    elif home_avg_goals > away_avg_goals + 0.5:
        return f"{home_name} Double Chance"
    elif away_avg_goals > home_avg_goals + 0.5:
        return f"{away_name} Double Chance"
    else:
        return "Under 3.5 Goals"


# ---------- MESSAGE FORMAT ----------
def format_message(fx, odds, probs, home_form, away_form, pick):
    home = fx["teams"]["home"]["name"]
    away = fx["teams"]["away"]["name"]
    league = fx["_league_name"]
    kickoff_local = fx["_kickoff_dt"].strftime("%H:%M UTC")

    odds_str = f"{home} {odds['home']} • DRAW {odds['draw']} • {away} {odds['away']}" if odds else "N/A"
    ml_str = f"{home} {probs['home']}% • DRAW {probs['draw']}% • {away} {probs['away']}%" if probs else "N/A"

    return (
        f"⚽ *{home} vs {away}* — {kickoff_local} — {league}\n"
        f"• form_bot: {home[:3].upper()} {home_form} | {away[:3].upper()} {away_form}\n"
        f"• odds_bot: {odds_str}\n"
        f"• ml_bot: {ml_str}\n"
        f"• Combined Pick: {pick}\n"
        f"\n⏰ Kickoff in ~30 minutes"
    )


# ---------- TELEGRAM ----------
def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": CHANNEL_ID,
        "text": text,
        "parse_mode": "Markdown",
    })
    if not resp.ok:
        print("Telegram send failed:", resp.text)


# ---------- CORE CHECK (same as before) ----------
def check_and_send():
    sent = cleanup_old_sent(load_sent())
    now = datetime.now(timezone.utc)

    fixtures = fetch_fixtures_today()
    print(f"[{now.isoformat()}] Checked {len(fixtures)} fixtures across top 5 leagues.")

    for fx in fixtures:
        fixture_id = str(fx["fixture"]["id"])
        kickoff_dt = datetime.fromisoformat(fx["fixture"]["date"].replace("Z", "+00:00"))
        fx["_kickoff_dt"] = kickoff_dt

        mins_to_kickoff = (kickoff_dt - now).total_seconds() / 60

        if ALERT_WINDOW_MIN <= mins_to_kickoff <= ALERT_WINDOW_MAX and fixture_id not in sent:
            home_id = fx["teams"]["home"]["id"]
            away_id = fx["teams"]["away"]["id"]
            home_name = fx["teams"]["home"]["name"]
            away_name = fx["teams"]["away"]["name"]

            odds = fetch_odds(fx["fixture"]["id"])
            probs = implied_probabilities(odds)
            home_form, home_avg = fetch_last5_form(home_id)
            away_form, away_avg = fetch_last5_form(away_id)
            pick = build_combined_pick(home_avg, away_avg, home_name, away_name)

            message = format_message(fx, odds, probs, home_form, away_form, pick)
            send_telegram(message)
            print(f"Sent alert for {home_name} vs {away_name}")

            sent[fixture_id] = {"date": now.strftime("%Y-%m-%d")}

    save_sent(sent)


# ---------- BACKGROUND LOOP ----------
def background_loop():
    while True:
        try:
            check_and_send()
        except Exception as e:
            print("Error during check_and_send:", e)
        time.sleep(CHECK_INTERVAL_SECONDS)


# ---------- FLASK APP (keeps Render Web Service alive) ----------
app = Flask(__name__)

@app.route("/")
def home():
    return "Top 5 Match Pick bot is running.", 200


# Start the background loop once, when the app starts
threading.Thread(target=background_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
