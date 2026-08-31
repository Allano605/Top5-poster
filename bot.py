import os
from threading import Thread
from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
flask_app = Flask(__name__)

@flask_app.route('/')
def home():
    return "Bot is Live"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot Live! Send /top5")

async def top5(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("TOP 5:\n1) Game 1\n2) Game 2\n3) Game 3\n4) Game 4\n5) Game 5")

def run_bot():
    if not TOKEN:
        print("BOT_TOKEN not set!")
        return
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("top5", top5))
    app.run_polling()

if __name__ == "__main__":
    Thread(target=run_bot).start()
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host="0.0.0.0", port=port)
