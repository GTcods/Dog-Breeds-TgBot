from typing import Final
from telegram import Update
import tensorflow as tf
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from Prediction.prediction import predict_breed, preprocess_image
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import os
import datetime

load_dotenv()  # Load environment variables from .env file
TOKEN = os.getenv('USER_TOKEN')
BOT_USERNAME: Final = '@doggy_breeds_bot'


model1 = tf.keras.models.load_model('Prediction/dog_breed_classifier_2.keras')
model2 = tf.keras.models.load_model('Prediction/dog_breed_classifier_10.keras')
model3 = tf.keras.models.load_model('Prediction/dog_breed_classifier.keras')

user_sessions = {}


async def create_user_session(user_id):
    user_sessions[user_id] = {
        'current_state': 'waiting_for_input',
        'previous_inputs': [],
        'model': model1,
        'version': 1
        }
    print(f'{len(user_sessions)} Users')
    print(user_sessions)


async def update_user_session(user_id, key, value):
    if user_id in user_sessions:
        user_sessions[user_id][key] = value


async def get_user_session(user_id):
    return user_sessions.get(user_id, None)


# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_sessions:
        await create_user_session(user_id)
    await update.message.reply_text("Hi, Select the model and start predicting!")


async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    await update_user_session(user_id, 'model', model1)
    await update_user_session(user_id, 'version', 1)

    await update.message.reply_text("Restarting...")


async def first_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    print(f'{user_id} User picked model 1', datetime.datetime.now())

    await update_user_session(user_id, 'model', model1, )
    await update_user_session(user_id, 'version', 1)

    await update.message.reply_text("Model 1 Selected!")


async def second_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    print(f'{user_id} User picked model 2', datetime.datetime.now())

    await update_user_session(user_id, 'model', model2)
    await update_user_session(user_id, 'version', 2)

    await update.message.reply_text("Model 2 Selected!")


async def final_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    print(f'{user_id} User picked final model', datetime.datetime.now())

    await update_user_session(user_id, 'model', model3)
    await update_user_session(user_id, 'version', 3)

    await update.message.reply_text("Final Model Selected!")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Helping...")


async def end_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("The End...")


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
    user_id = update.effective_user.id
    print(user_id, datetime.datetime.now())

    photo_file = await update.message.photo[-1].get_file()  # Get the file object
    photo_bytes = await photo_file.download_as_bytearray()  # Download as byte array

    photo_stream = BytesIO(photo_bytes)  # Create BytesIO object
    # Open the image using PIL and convert to RGB
    img = Image.open(photo_stream).convert('RGB')  # Ensure it's in RGB format
    img_array = preprocess_image(img)  # Preprocess the image

    breed_name = predict_breed(user_sessions[user_id]['model'], img_array, user_sessions[user_id]['version'])

    await update.message.reply_text(f"The predicted breed is: {breed_name}")


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


if __name__ == '__main__':
    print("Starting bot...")
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('restart', restart_command))
    app.add_handler(CommandHandler('first', first_command))
    app.add_handler(CommandHandler('second', second_command))
    app.add_handler(CommandHandler('final', final_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('end', end_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Errors
    app.add_error_handler(error)

    print('Polling...')
    app.run_polling(poll_interval=3)
