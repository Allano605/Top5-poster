"""
Top 5 Match Pick — Auto Alert Bot (with trained ML model for Over/Under 2.5)
------------------------------------------------------------------------------
Same as before: pulls TODAY's fixtures from API-Football for the top 5
leagues, checks kickoff timing, sends a Telegram alert 30 min before each
match. form_bot and odds_bot are unchanged.
"""

import os
import json
import time
import threading
import requests
import joblib
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request

# ---------- CONFIG ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
CHANNEL_ID = os.environ["CHANNEL_ID"]
API_KEY = os.environ["FOOTBALL_API_KEY"]

AFFILIATE_LINK = os.environ.get("AFFILIATE_LINK", "")

API_HOST = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}
LEAGUE_NAME_TO_MODEL = {
    "Premier League": "League_E0",
    "La Liga": "League_SP1",
    "Serie A": "League_I1",
    "Bundesliga": "League_D1",
    "Ligue 1": "League_F1",
}

SENT_FILE = "sent_matches.json"
ALERT_WINDOW_MIN = 25
ALERT_WINDOW_MAX = 35
CHECK_INTERVAL_SECONDS = 300

PENDING_FILE = "pending_subs.json"
FLW_SECRET_HASH = os.environ.get("FLW_SECRET_HASH", "")

TIER_CHANNELS = {
    "silver": os.environ.get("SILVER_CHANNEL_ID", ""),
    "gold": os.environ.get("GOLD_CHANNEL_ID", ""),
    "vip": os.environ.get("VIP_CHANNEL_ID", ""),
}
TIER_PAY_LINKS = {
    "silver": os.environ.get("SILVER_PAY_LINK", ""),
    "gold": os.environ.get("GOLD_PAY_LINK", ""),
    "vip": os.environ.get("VIP_PAY_LINK", ""),
}
TIER_PRICES = {"silver": "N3,000/month", "gold": "N6,000/month", "vip": "N10,000/month"}

def load_pending():
    if os.path.exists(PENDING_FILE):
        with open(PENDING_FILE, "r") as f:
            return json.load(f)
    return {}

def save_pending(data):
    with open(PENDING_FILE, "w") as f:
        json.dump(data, f)

ML_MODEL = joblib.load("over25_model.pkl")
ML_FEATURE_COLS = joblib.load("over25_model_features.pkl")
print(f"Loaded ML model with {len(ML_FEATURE_COLS)} features.")

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
    resp = requests.get(f"{API_HOST}/odds", headers=HEADERS, params={"fixture": fixture_id})
    data = resp.json().get("response", [])
    if not data:
        return None
    try:
        bookmaker = data[0]["bookmakers"][0]
        bets = bookmaker["bets"][0]["values"]
        odds = {v["value"]: float(v["odd"]) for v in bets}
        return {"home": odds.get("Home"), "draw": odds.get("Draw"), "away": odds.get("Away")}
    except (KeyError, IndexError):
        return None

def fetch_recent_matches(team_id, last_n=10):
    resp = requests.get(
        f"{API_HOST}/fixtures",
        headers=HEADERS,
        params={"team": team_id, "last": last_n},
    )
    return resp.json().get("response", [])

def compute_rolling_stats(matches, team_id):
    goals_for, goals_against = [], []
    dates = []
    for fx in matches:
        teams = fx["teams"]
        goals = fx["goals"]
        is_home = teams["home"]["id"] == team_id
        gf = goals["home"] if is_home else goals["away"]
        ga = goals["away"] if is_home else goals["home"]
        if gf is not None and ga is not None:
            goals_for.append(gf)
            goals_against.append(ga)
            dates.append(fx["fixture"]["date"])

    if not goals_for:
        return None

    avg5_goals_for = np.mean(goals_for[-5:]) if len(goals_for) >= 3 else np.mean(goals_for)
    avg5_goals_against = np.mean(goals_against[-5:]) if len(goals_against) >= 3 else np.mean(goals_against)
    last3 = np.mean(goals_for[-3:]) if len(goals_for) >= 2 else avg5_goals_for
    last10 = np.mean(goals_for[-10:]) if len(goals_for) >= 5 else avg5_goals_for
    trend = last3 - last10

    rest_days = None
    if len(dates) >= 1:
        last_date = datetime.fromisoformat(dates[-1].replace("Z", "+00:00"))
        rest_days = (datetime.now(timezone.utc) - last_date).days

    return {
        "goals_for_avg5": avg5_goals_for,
        "goals_against_avg5": avg5_goals_against,
        "trend": trend,
        "rest_days": rest_days if rest_days is not None else 7,
    }

def build_ml_features(home_id, away_id, league_name, home_matches, away_matches):
    home_stats = compute_rolling_stats(home_matches, home_id)
    away_stats = compute_rolling_stats(away_matches, away_id)

    if not home_stats or not away_stats:
        return None

    row = {col: 0.0 for col in ML_FEATURE_COLS}

    if "Home_GoalsFor_avg5" in row:
        row["Home_GoalsFor_avg5"] = home_stats["goals_for_avg5"]
    if "Home_GoalsAgainst_avg5" in row:
        row["Home_GoalsAgainst_avg5"] = home_stats["goals_against_avg5"]
    if "Away_GoalsFor_avg5" in row:
        row["Away_GoalsFor_avg5"] = away_stats["goals_for_avg5"]
    if "Away_GoalsAgainst_avg5" in row:
        row["Away_GoalsAgainst_avg5"] = away_stats["goals_against_avg5"]
    if "Home_GoalsFor_trend" in row:
        row["Home_GoalsFor_trend"] = home_stats["trend"]
    if "Away_GoalsFor_trend" in row:
        row["Away_GoalsFor_trend"] = away_stats["trend"]
    if "Home_RestDays" in row:
        row["Home_RestDays"] = home_stats["rest_days"]
    if "Away_RestDays" in row:
        row["Away_RestDays"] = away_stats["rest_days"]

    league_col = LEAGUE_NAME_TO_MODEL.get(league_name)
    if league_col and league_col in row:
        row[league_col] = 1.0

    feature_vector = [row.get(col, 0.0) for col in ML_FEATURE_COLS]
    return feature_vector

def predict_over25_probability(home_id, away_id, league_name):
    try:
        home_matches = fetch_recent_matches(home_id)
        away_matches = fetch_recent_matches(away_id)
        features = build_ml_features(home_id, away_id, league_name, home_matches, away_matches)
        if features is None:
            return None
        prob = ML_MODEL.predict_proba([features])[0][1]
        return round(prob * 100)
    except Exception as e:
        print("ML prediction failed:", e)
        return None

def implied_probabilities(odds):
    if not odds or not all([odds.get("home"), odds.get("draw"), odds.get("away")]):
        return None
    raw = {k: 1 / v for k, v in odds.items()}
    total = sum(raw.values())
    return {k: round((v / total) * 100) for k, v in raw.items()}

def fetch_last5_form(team_id):
    resp = requests.get(f"{API_HOST}/fixtures", headers=HEADERS, params={"team": team_id, "last": 5})
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

def build_combined_pick(home_avg_goals, away_avg_goals, home_name, away_name, ml_over25_pct):
    total_avg = home_avg_goals + away_avg_goals
    if ml_over25_pct is not None and ml_over25_pct >= 60:
        return "Over 2.5 Goals (ML-supported)"
    if home_avg_goals >= 1 and away_avg_goals >= 1:
        return "Both Teams to Score - Yes"
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

def format_message(fx, odds, probs, home_form, away_form, pick, ml_over25_pct):
    home = fx["teams"]["home"]["name"]
    away = fx["teams"]["away"]["name"]
    league = fx["_league_name"]
    kickoff_local = fx["_kickoff_dt"].strftime("%H:%M UTC")

    odds_str = f"{home} {odds['home']} - DRAW {odds['draw']} - {away} {odds['away']}" if odds else "N/A"

    if ml_over25_pct is not None:
        ml_str = f"Over 2.5 Goals: {ml_over25_pct}% (trained model, ~58% backtested accuracy)"
    else:
        ml_str = "N/A (insufficient recent data)"

    message = (
        f"{home} vs {away} - {kickoff_local} - {league}\n"
        f"form_bot: {home[:3].upper()} {home_form} | {away[:3].upper()} {away_form}\n"
        f"odds_bot: {odds_str}\n"
        f"ml_bot: {ml_str}\n"
        f"Combined Pick: {pick}\n"
        f"\nKickoff in ~30 minutes"
    )

    if AFFILIATE_LINK:
        message += f"\n\nPlace your bet: {AFFILIATE_LINK}\n18+ | Bet responsibly"

    return message

def send_telegram(text, chat_id=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id or CHANNEL_ID, "text": text, "parse_mode": "Markdown"})
    if not resp.ok:
        print("Telegram send failed:", resp.text)

def create_invite_link(channel_id):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/createChatInviteLink"
    resp = requests.post(url, data={"chat_id": channel_id, "member_limit": 1})
    data = resp.json()
    if data.get("ok"):
        return data["result"]["invite_link"]
    print("Invite link creation failed:", data)
    return None

def handle_incoming_message(update):
    message = update.get("message", {})
    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id")

    if not text or not chat_id:
        return

    if text.startswith("/subscribe"):
        parts = text.strip().split()
        tier = parts[1].lower() if len(parts) > 1 else None

        if tier not in TIER_PAY_LINKS or not TIER_PAY_LINKS[tier]:
            send_telegram(
                "Choose a tier: /subscribe silver, /subscribe gold, or /subscribe vip",
                chat_id=chat_id,
            )
            return

        send_telegram(
            f"{tier.upper()} - {TIER_PRICES[tier]}\n\n"
            f"1. Pay here: {TIER_PAY_LINKS[tier]}\n"
            f"2. Use this exact code as your payment description/note: {chat_id}\n"
            f"3. Once payment confirms, you'll automatically get an invite link here - no waiting.",
            chat_id=chat_id,
        )

        pending = load_pending()
        pending[str(chat_id)] = {"tier": tier, "chat_id": chat_id}
        save_pending(pending)

    elif text.startswith("/start"):
        send_telegram(
            "Welcome to Top 5 Match Pick!\n\n"
            "Type /subscribe silver, /subscribe gold, or /subscribe vip to unlock more daily analysis.",
            chat_id=chat_id,
        )

def handle_flutterwave_webhook(payload):
    try:
        data = payload.get("data", {})
        status = data.get("status")
        narration = str(data.get("narration", "") or data.get("meta", {}).get("chat_id", ""))

        if status != "successful":
            return

        pending = load_pending()
        matched_chat_id = None
        for key, entry in pending.items():
            if str(entry["chat_id"]) in narration:
                matched_chat_id = key
                break

        if not matched_chat_id:
            print("No matching pending subscription found for this payment.")
            return

        entry = pending[matched_chat_id]
        tier = entry["tier"]
        channel_id = TIER_CHANNELS.get(tier)

        if not channel_id:
            print(f"No channel configured for tier: {tier}")
            return

        invite_link = create_invite_link(channel_id)
        if invite_link:
            send_telegram(
                f"Payment confirmed! Here's your {tier.upper()} channel invite:\n{invite_link}\n\n"
                f"(Link works once, for you only.)",
                chat_id=entry["chat_id"],
            )
            del pending[matched_chat_id]
            save_pending(pending)
    except Exception as e:
        print("Error handling Flutterwave webhook:", e)

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
            league_name = fx["_league_name"]

            odds = fetch_odds(fx["fixture"]["id"])
            probs = implied_probabilities(odds)
            home_form, home_avg = fetch_last5_form(home_id)
            away_form, away_avg = fetch_last5_form(away_id)

            ml_over25_pct = predict_over25_probability(home_id, away_id, league_name)
            pick = build_combined_pick(home_avg, away_avg, home_name, away_name, ml_over25_pct)

            message = format_message(fx, odds, probs, home_form, away_form, pick, ml_over25_pct)
            send_telegram(message)
            print(f"Sent alert for {home_name} vs {away_name} (ML Over2.5: {ml_over25_pct}%)")

            sent[fixture_id] = {
                "date": now.strftime("%Y-%m-%d"),
                "home": home_name,
                "away": away_name,
                "pick": pick,
                "kickoff": kickoff_dt.isoformat(),
                "result_posted": False,
            }

    save_sent(sent)

def fetch_fixture_result(fixture_id):
    resp = requests.get(f"{API_HOST}/fixtures", headers=HEADERS, params={"id": fixture_id})
    data = resp.json().get("response", [])
    if not data:
        return None
    fx = data[0]
    status = fx["fixture"]["status"]["short"]
    if status != "FT":
        return None
    home_goals = fx["goals"]["home"]
    away_goals = fx["goals"]["away"]
    if home_goals is None or away_goals is None:
        return None
    return home_goals, away_goals

def pick_won(pick, home_goals, away_goals):
    total = home_goals + away_goals
    if "Over 2.5" in pick:
        return total > 2.5
    if "Under" in pick:
        return total < 3.5
    if "Both Teams to Score" in pick:
        return home_goals >= 1 and away_goals >= 1
    return None

def check_results_and_post_wins():
    sent = load_sent()
    now = datetime.now(timezone.utc)
    changed = False

    for fixture_id, entry in sent.items():
        if entry.get("result_posted"):
            continue
        if "kickoff" not in entry:
            continue

        kickoff_dt = datetime.fromisoformat(entry["kickoff"])
        if (now - kickoff_dt).total_seconds() < 7200:
            continue

        result = fetch_fixture_result(fixture_id)
        if result is None:
            continue

        home_goals, away_goals = result
        won = pick_won(entry["pick"], home_goals, away_goals)
        entry["result_posted"] = True
        changed = True

        if won:
            caption = (
                f"Free tier caught this one 👀\n"
                f"{entry['home']} {home_goals}-{away_goals} {entry['away']} — {entry['pick']} ✅\n"
                f"Gold members ALSO got tomorrow's picks already loaded.\n"
                f"No be everybody go see am first though."
            )
            send_telegram(caption)
            print(f"Posted win for {entry['home']} vs {entry['away']}")

    if changed:
        save_sent(sent)

def background_loop():
    while True:
        try:
            check_and_send()
            check_results_and_post_wins()
        except Exception as e:
            print("Error during background_loop:", e)
        time.sleep(CHECK_INTERVAL_SECONDS)

app = Flask(__name__)

@app.route("/")
def home():
    return "Top 5 Match Pick bot is running (with trained ML model).", 200

@app.route("/telegram-webhook", methods=["POST"])
def telegram_webhook():
    update = request.get_json(force=True, silent=True) or {}
    try:
        handle_incoming_message(update)
    except Exception as e:
        print("Error handling Telegram update:", e)
    return "OK", 200

@app.route("/flutterwave-webhook", methods=["POST"])
def flutterwave_webhook():
    signature = request.headers.get("verif-hash", "")
    if not FLW_SECRET_HASH or signature != FLW_SECRET_HASH:
        print("Flutterwave webhook: signature mismatch, ignoring.")
        return "Unauthorized", 401

    payload = request.get_json(force=True, silent=True) or {}
    handle_flutterwave_webhook(payload)
    return "OK", 200

threading.Thread(target=background_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
