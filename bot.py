import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO)
BOT_TOKEN = os.getenv("BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚽ Top5 Poster Bot is LIVE!\n\nUse /top5 to test.")

async def top5(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔥 TOP 5 LEAGUES\n1. Premier League\n2. La Liga\n3. Bundesliga\n4. Serie A\n5. Ligue 1")

def main():
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN missing")
        return
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("top5", top5))
    print("Bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
