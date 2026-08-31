import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Setup logging
logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Use /top5 to get top 5.")

async def top5(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = "🏆 TOP 5:\n\n1. Item 1\n2. Item 2\n3. Item 3\n4. Item 4\n5. Item 5"
    await update.message.reply_text(message)

def main():
    if not TOKEN:
        raise ValueError("BOT_TOKEN not set!")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("top5", top5))
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
