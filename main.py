# main.py
import logging
import io
import qrcode
import asyncio
import os.path

from openai import OpenAI
# import google.generativeai as genai # معطل مؤقتًا

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler
)

# استيرادات Google Drive API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# --- إعدادات البوت الأساسية ---
BOT_TOKEN = "7719226402:AAHQ2RMk5e4SFDBhaVxk-hodY9PelRjRVFY"

# --- إعدادات مفاتيح API ---
OPENAI_API_KEY = "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"

ADMIN_ID_1 = 1263152179
ADMIN_ID_2 = 697852646
ADMIN_IDS = [ADMIN_ID_1, ADMIN_ID_2]

# --- إعدادات Google Drive API ---
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']
CLIENT_SECRET_FILE = 'client_secret.json'
TOKEN_FILE = 'token.json'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- دوال Google Drive ---
async def get_gdrive_service_async():
    def _authenticate_gdrive():
        creds = None
        logger.info(f"DEBUG_AUTH: Current Working Directory in _authenticate_gdrive: {os.getcwd()}")
        logger.info(f"DEBUG_AUTH: Checking for TOKEN_FILE at: {os.path.join(os.getcwd(), TOKEN_FILE)}")
        logger.info(f"DEBUG_AUTH: TOKEN_FILE exists? {os.path.exists(os.path.join(os.getcwd(), TOKEN_FILE))}")
        logger.info(f"DEBUG_AUTH: Checking for CLIENT_SECRET_FILE at: {os.path.join(os.getcwd(), CLIENT_SECRET_FILE)}")
        logger.info(f"DEBUG_AUTH: CLIENT_SECRET_FILE exists? {os.path.exists(os.path.join(os.getcwd(), CLIENT_SECRET_FILE))}")

        if os.path.exists(TOKEN_FILE):
            try:
                creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            except Exception as e:
                logger.error(f"Error loading token file '{TOKEN_FILE}': {e}. Will attempt to re-authenticate.")
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Attempting to refresh Google Drive token...")
                    creds.refresh(Request())
                    logger.info("Google Drive token refreshed successfully.")
                except Exception as e:
                    logger.error(f"Failed to refresh Google Drive token: {e}. Need to re-authenticate.")
                    creds = None
            
            if not creds:
                try:
                    logger.info(f"'{TOKEN_FILE}' not found or invalid. Starting new authentication flow using '{CLIENT_SECRET_FILE}'.")
                    if not os.path.exists(CLIENT_SECRET_FILE):
                        logger.error(f"Critical: Credentials file '{CLIENT_SECRET_FILE}' not found at path: {os.path.join(os.getcwd(), CLIENT_SECRET_FILE)}. Cannot start auth flow.")
                        return None
                    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
                    # !!! تم التعديل هنا لاستخدام run_local_server !!!
                    creds = flow.run_local_server(port=0) 
                    logger.info("Google Drive authentication flow completed via local server.")
                except FileNotFoundError:
                    logger.error(f"Credentials file '{CLIENT_SECRET_FILE}' was not found by InstalledAppFlow. Path: {os.path.join(os.getcwd(), CLIENT_SECRET_FILE)}")
                    return None
                except Exception as e:
                    logger.error(f"An error occurred during Google Drive authentication flow: {e}")
                    return None
            
            try:
                with open(TOKEN_FILE, 'w') as token_file:
                    token_file.write(creds.to_json())
                logger.info(f"Google Drive token saved to {TOKEN_FILE}")
            except Exception as e:
                logger.error(f"Failed to save Google Drive token to {TOKEN_FILE}: {e}")

        try:
            service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive service object created successfully.")
            return service
        except HttpError as error:
            logger.error(f'An HttpError occurred while building Google Drive service: {error.resp.status} - {error.content}')
            if error.resp.status in [401, 403]:
                logger.warning("Received 401/403 error. Deleting old token file to force re-authentication.")
                if os.path.exists(TOKEN_FILE):
                    try:
                        os.remove(TOKEN_FILE)
                        logger.info(f"Deleted {TOKEN_FILE}. Please try the command again to re-authenticate.")
                    except Exception as e_remove:
                        logger.error(f"Could not delete token file {TOKEN_FILE}: {e_remove}")
            return None
        except Exception as e:
            logger.error(f'An unexpected error occurred while building Google Drive service: {e}')
            return None

    return await asyncio.to_thread(_authenticate_gdrive)

# --- (بقية دوال معالجة الأوامر والرسائل الأساسية كما هي) ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome_message = (
        f"أهلاً بك يا {user.first_name}!\n"
        f"أنا بوتك المساعد. جرب الأوامر:\n"
        f"/qr <نص> - لإنشاء QR Code\n"
        f"/testai <سؤال> - لطرح سؤال على OpenAI\n"
        f"/gdrivefiles - لعرض أول 5 ملفات من Google Drive\n"
        f"أو أرسل لي أي رسالة نصية وسأرددها."
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)
    logger.info(f"User {user.id} ({user.first_name}) started the bot.")

async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    reply_text = f"أنت قلت: {user_message}"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=reply_text)
    logger.info(f"User {update.effective_user.id} sent text: '{user_message}', bot echoed.")

async def admin_test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in ADMIN_IDS:
        reply_text = f"أهلاً بالمسؤول {update.effective_user.first_name}! هذا أمر خاص بك."
    else:
        reply_text = "عذراً، هذا الأمر مخصص للمسؤولين فقط."
    await context.bot.send_message(chat_id=update.effective_chat.id, text=reply_text)

async def generate_qr_image(text_to_encode: str) -> io.BytesIO | None:
    if not text_to_encode:
        return None
    qr_img = qrcode.make(text_to_encode)
    img_byte_arr = io.BytesIO()
    qr_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

async def qr_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("لاستخدام الأمر، أرسل: /qr <النص أو الرابط>")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
    text_to_encode = " ".join(context.args)
    qr_image_stream = await generate_qr_image(text_to_encode)
    if qr_image_stream:
        await update.message.reply_photo(photo=qr_image_stream, caption=f"QR Code لـ: {text_to_encode}")
    else:
        await update.message.reply_text("خطأ في إنشاء QR Code. تأكد من إدخال نص.")

async def get_openai_response(api_key: str, user_question: str) -> str:
    if not api_key or len(api_key) < 50:
        logger.error("OpenAI API key is not configured or seems too short.")
        return "عذراً، لم يتم إعداد مفتاح OpenAI API بشكل صحيح."
    try:
        logger.info(f"Sending request to OpenAI API with question: {user_question}")
        def generate_sync():
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Please respond in Arabic unless the user asks for another language."},
                    {"role": "user", "content": user_question}
                ],
                max_tokens=300
            )
            return completion.choices[0].message.content.strip()
        assistant_reply = await asyncio.to_thread(generate_sync)
        logger.info("Received response from OpenAI API.")
        return assistant_reply if assistant_reply else "لم أتلق ردًا نصيًا من OpenAI."
    except Exception as e:
        logger.error(f"Error communicating with OpenAI API: {e}")
        error_message = str(e).lower()
        if "invalid api key" in error_message or "incorrect api key" in error_message:
             return "عذراً، مفتاح OpenAI API غير صالح. يرجى التحقق منه."
        if "quota" in error_message or "rate limit" in error_message:
            return "عذراً، لقد تجاوزت حصة الاستخدام أو الحد الأقصى للطلبات لـ OpenAI API. يرجى المحاولة لاحقًا."
        return f"عذراً، حدث خطأ أثناء محاولة الاتصال بـ OpenAI: ({type(e).__name__})"

async def testai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("لاستخدام الأمر، أرسل: /testai <سؤالك لـ OpenAI>")
        return
    user_question = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    thinking_message = await update.message.reply_text("لحظات، جاري التواصل مع OpenAI... 🧠")
    logger.info(f"User {update.effective_user.id} asked OpenAI (via /testai): '{user_question}'")
    if not OPENAI_API_KEY or len(OPENAI_API_KEY) < 50:
        reply_text = "مفتاح OpenAI API غير مُعد بشكل صحيح. يرجى الاتصال بمسؤول البوت."
        final_markup = None
        logger.error("OpenAI API key is a placeholder or missing in testai_command.")
    else:
        reply_text = await get_openai_response(OPENAI_API_KEY, user_question)
        keyboard = [
            [
                InlineKeyboardButton("👍 إجابة مفيدة", callback_data='feedback_useful'),
                InlineKeyboardButton("👎 غير مفيدة", callback_data='feedback_not_useful'),
            ]
        ]
        final_markup = InlineKeyboardMarkup(keyboard)
    try:
        await context.bot.edit_message_text(
            chat_id=thinking_message.chat_id,
            message_id=thinking_message.message_id,
            text=reply_text,
            reply_markup=final_markup
        )
    except Exception as e:
        logger.error(f"Error editing 'thinking' message: {e}. Sending new message instead.")
        await update.message.reply_text(reply_text, reply_markup=final_markup)
    logger.info("Sent OpenAI's response to user with feedback buttons.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer(text="شكرًا لتقييمك!")
    user_id = query.from_user.id
    logger.info(f"User {user_id} pressed button with data: {query.data}")
    if query.data == 'feedback_useful':
        await query.edit_message_text(text=f"{query.message.text}\n\n---\nتقييمك: إجابة مفيدة 👍")
    elif query.data == 'feedback_not_useful':
        await query.edit_message_text(text=f"{query.message.text}\n\n---\nتقييمك: إجابة غير مفيدة 👎")

async def list_gdrive_files_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    current_working_directory = os.getcwd()
    client_secret_path_to_check = os.path.join(current_working_directory, CLIENT_SECRET_FILE)
    token_file_path_to_check = os.path.join(current_working_directory, TOKEN_FILE)

    logger.info(f"DEBUG: Current Working Directory: {current_working_directory}")
    logger.info(f"DEBUG: Checking for CLIENT_SECRET_FILE ('{CLIENT_SECRET_FILE}') at: {client_secret_path_to_check}")
    logger.info(f"DEBUG: CLIENT_SECRET_FILE exists? {os.path.exists(client_secret_path_to_check)}")
    logger.info(f"DEBUG: Checking for TOKEN_FILE ('{TOKEN_FILE}') at: {token_file_path_to_check}")
    logger.info(f"DEBUG: TOKEN_FILE exists? {os.path.exists(token_file_path_to_check)}")
    
    if not os.path.exists(client_secret_path_to_check):
        await update.message.reply_text(
            f"ملف `{CLIENT_SECRET_FILE}` غير موجود في المسار المتوقع: `{current_working_directory}`. "
            f"يرجى التأكد من وضعه في مجلد المشروع."
        )
        logger.error(f"Missing {CLIENT_SECRET_FILE} at {client_secret_path_to_check}")
        return

    if not os.path.exists(token_file_path_to_check):
        await context.bot.send_message(chat_id, 
            "للوصول إلى Google Drive، أحتاج إلى إذنك لأول مرة. "
            "البوت سيحاول الآن فتح صفحة مصادقة في متصفحك الافتراضي. " # رسالة جديدة
            "يرجى اتباع التعليمات في المتصفح للموافقة على الأذونات."
            "\nقد تحتاج للقيام بهذه الخطوة مرة واحدة فقط."
        )

    service = await get_gdrive_service_async()

    if not service:
        await update.message.reply_text(
            "لم أتمكن من الاتصال بخدمة Google Drive. "
            "قد تحتاج لإعادة المصادقة أو التحقق من ملف `client_secret.json` والسجلات (logs)."
        )
        return

    try:
        logger.info(f"User {update.effective_user.id} requested to list Google Drive files.")
        
        def list_files_sync():
            return service.files().list(
                pageSize=5, 
                fields="nextPageToken, files(id, name, mimeType,webViewLink)",
                orderBy="modifiedTime desc"
            ).execute()

        results = await asyncio.to_thread(list_files_sync)
        items = results.get('files', [])

        if not items:
            await update.message.reply_text('لم يتم العثور على ملفات في Google Drive (أو ليس لدي إذن للوصول).')
        else:
            message_lines = ['أحدث 5 ملفات/مجلدات في Google Drive:\n']
            for item in items:
                icon = "📁" if item['mimeType'] == 'application/vnd.google-apps.folder' else "📄"
                link = item.get('webViewLink', '')
                name_escaped = item['name'].replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)').replace('~', '\\~').replace('`', '\\`').replace('>', '\\>').replace('#', '\\#').replace('+', '\\+').replace('-', '\\-').replace('=', '\\=').replace('|', '\\|').replace('{', '\\{').replace('}', '\\}').replace('.', '\\.').replace('!', '\\!')
                if link:
                    message_lines.append(f"{icon} [{name_escaped}]({link})")
                else:
                    message_lines.append(f"{icon} {name_escaped}")
            
            await update.message.reply_text("\n".join(message_lines), parse_mode='MarkdownV2')

    except HttpError as error:
        logger.error(f'An HttpError occurred with Google Drive API: {error.resp.status} - {error.content}')
        await update.message.reply_text(f'حدث خطأ أثناء الاتصال بـ Google Drive. رمز الخطأ: {error.resp.status}')
    except Exception as e:
        logger.error(f'An unexpected error occurred while listing files: {e}')
        await update.message.reply_text(f'حدث خطأ غير متوقع أثناء محاولة عرض الملفات: {type(e).__name__}')


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="عذراً، لم أفهم هذا الأمر.")

# --- الدالة الرئيسية لتشغيل البوت ---
if __name__ == '__main__':
    logger.info("Starting bot with GDrive, QR, and OpenAI features...")

    application = ApplicationBuilder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler('start', start_command))
    application.add_handler(CommandHandler('admin_test', admin_test_command))
    application.add_handler(CommandHandler('qr', qr_command_handler))
    application.add_handler(CommandHandler('testai', testai_command))
    application.add_handler(CommandHandler('gdrivefiles', list_gdrive_files_command))

    application.add_handler(CallbackQueryHandler(button_callback))

    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo_message))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    logger.info("Bot is now polling for updates...")
    application.run_polling()

    logger.info("Bot has stopped.")