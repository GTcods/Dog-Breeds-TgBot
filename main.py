from typing import Final
from telegram import Update
import tensorflow as tf
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from Prediction.prediction import predict_breed, preprocess_image
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
TOKEN = os.getenv('USER_TOKEN')
BOT_USERNAME: Final = '@doggy_breeds_bot'

model = tf.keras.models.load_model('Prediction/dog_breed_classifier.keras')


# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Help")


async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Custom")


# Responses
def handle_response(text: str):
    processed_text = text.lower()

    if 'hello' in processed_text:
        return "Hi there!"

    if 'mariam' in processed_text:
        return "Pspspsss..."

    return "Try sending Pictures"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f"User ({update.message.chat.id}) in {message_type}: '{text}'")

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, "").strip()
            response = handle_response(new_text)
        else:
            return

    else:
        response: str = handle_response(text)

    print('Bot:', response)
    await update.message.reply_text(response)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()  # Get the file object
    photo_bytes = await photo_file.download_as_bytearray()  # Download as byte array

    photo_stream = BytesIO(photo_bytes)  # Create BytesIO object

    # Open the image using PIL and convert to RGB
    img = Image.open(photo_stream).convert('RGB')  # Ensure it's in RGB format

    img_array = preprocess_image(img)  # Preprocess the image

    breed_name = predict_breed(model, img_array)

    await update.message.reply_text(f"The predicted breed is: {breed_name}")


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


if __name__ == '__main__':
    print("Starting bot...")
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom_command', custom_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Errors
    app.add_error_handler(error)

    print('Polling...')
    app.run_polling(poll_interval=3)
