# main.py
import logging
import io
import qrcode
import asyncio
import os
from dotenv import load_dotenv # <-- لاستيراد متغيرات البيئة من .env
import pandas as pd # <-- لاستيراد pandas

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

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
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ADMIN_ID_1 = 1263152179
ADMIN_ID_2 = 697852646
ADMIN_IDS = [ADMIN_ID_1, ADMIN_ID_2]

# --- إعدادات Google Drive API ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CLIENT_SECRET_FILE = 'client_secret.json'
TOKEN_FILE = 'token.json'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO # يمكنك إعادته إلى INFO الآن أو تركه DEBUG إذا أردت المزيد من التفاصيل
)
logger = logging.getLogger(__name__)
# logging.getLogger("httpx").setLevel(logging.WARNING) # يمكنك إعادة تفعيل هذا لتقليل ضوضاء httpx
# logging.getLogger("telegram.ext").setLevel(logging.DEBUG) # يمكنك ترك هذا DEBUG أو إعادته لـ INFO

if not BOT_TOKEN:
    logger.critical("CRITICAL: BOT_TOKEN is not set. The bot cannot start.")
    exit()
if not OPENAI_API_KEY:
    logger.warning("!!! OPENAI_API_KEY not set. OpenAI features will not work.")

# --- دوال Google Drive ---
async def get_gdrive_service_async():
    def _authenticate_gdrive():
        creds = None
        if os.path.exists(TOKEN_FILE):
            try:
                creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            except Exception as e:
                logger.error(f"Error loading token file '{TOKEN_FILE}': {e}.")
                creds = None
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Attempting to refresh Google Drive token...")
                    creds.refresh(Request())
                    logger.info("Google Drive token refreshed successfully.")
                except Exception as e:
                    logger.error(f"Failed to refresh Google Drive token: {e}.")
                    creds = None
            if not creds:
                try:
                    logger.info(f"'{TOKEN_FILE}' not found or invalid. Starting auth flow.")
                    if not os.path.exists(CLIENT_SECRET_FILE):
                        logger.error(f"Critical: '{CLIENT_SECRET_FILE}' not found.")
                        return None
                    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
                    creds = flow.run_local_server(port=0) 
                    logger.info("Google Drive authentication flow completed.")
                except Exception as e:
                    logger.error(f"Error in Google Drive authentication flow: {e}")
                    return None
            try:
                with open(TOKEN_FILE, 'w') as token_file: token_file.write(creds.to_json())
                logger.info(f"Google Drive token saved to {TOKEN_FILE}")
            except Exception as e: logger.error(f"Failed to save Google Drive token: {e}")
        try:
            service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive service object created.")
            return service
        except Exception as e:
            logger.error(f'Error building Google Drive service: {e}')
            if hasattr(e, 'resp') and e.resp.status in [401, 403]:
                if os.path.exists(TOKEN_FILE): os.remove(TOKEN_FILE)
            return None
    return await asyncio.to_thread(_authenticate_gdrive)

# --- (بقية الدوال مثل start_command, echo_message, admin_test_command, generate_qr_image, qr_command_handler, get_openai_response, testai_command كما هي) ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome_message = (
        f"أهلاً بك يا {user.first_name}!\n"
        f"أنا بوتك المساعد. جرب الأوامر:\n"
        f"/qr <نص> - لإنشاء QR Code\n"
        f"/testai <سؤال> - لطرح سؤال على OpenAI\n"
        f"/gdrivefiles - لعرض ملفات Google Drive (والقراءة منها)\n"
        "This bot is being enhanced by Jules!\n"
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
    if not OPENAI_API_KEY:
        reply_text = "مفتاح OpenAI API غير مُعد في متغيرات البيئة."
        final_markup = None
        logger.error("OPENAI_API_KEY is not set.")
    else:
        reply_text = await get_openai_response(OPENAI_API_KEY, user_question)
        keyboard = [[ InlineKeyboardButton("👍", callback_data='feedback_useful'), InlineKeyboardButton("👎", callback_data='feedback_not_useful'),]]
        final_markup = InlineKeyboardMarkup(keyboard)
    try:
        await context.bot.edit_message_text(chat_id=thinking_message.chat_id, message_id=thinking_message.message_id, text=reply_text, reply_markup=final_markup)
    except Exception as e:
        logger.error(f"Error editing 'thinking' message: {e}.")
        await update.message.reply_text(reply_text, reply_markup=final_markup)
    logger.info("Sent OpenAI's response.")

# --- معالج أمر Google Drive ---
async def list_gdrive_files_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
    if not os.path.exists(CLIENT_SECRET_FILE):
        await update.message.reply_text(f"ملف `{CLIENT_SECRET_FILE}` غير موجود في: `{os.getcwd()}`.")
        logger.error(f"Missing {CLIENT_SECRET_FILE} at {os.getcwd()}")
        return

    if not os.path.exists(TOKEN_FILE):
        await context.bot.send_message(chat_id, "للوصول لـ Google Drive، أحتاج إذنك (مرة واحدة). اتبع التعليمات في الطرفية.")

    service = await get_gdrive_service_async()
    if not service:
        await update.message.reply_text("لم أتمكن من الاتصال بـ Google Drive.")
        return

    try:
        logger.info(f"User {update.effective_user.id} requested GDrive files.")
        def list_files_sync():
            return service.files().list(pageSize=10, fields="files(id, name, mimeType,webViewLink)", orderBy="modifiedTime desc").execute()
        results = await asyncio.to_thread(list_files_sync)
        items = results.get('files', [])
        if not items:
            await update.message.reply_text('لا توجد ملفات في Google Drive.')
            return
        await update.message.reply_text("أحدث الملفات/المجلدات (ملفات CSV لها زر للقراءة):")
        for item in items:
            icon = "📁" if item['mimeType'] == 'application/vnd.google-apps.folder' else "📄"
            link = item.get('webViewLink', '')
            name_escaped = item['name'].replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)').replace('~', '\\~').replace('`', '\\`').replace('>', '\\>').replace('#', '\\#').replace('+', '\\+').replace('-', '\\-').replace('=', '\\=').replace('|', '\\|').replace('{', '\\{').replace('}', '\\}').replace('.', '\\.').replace('!', '\\!')
            file_info_line = f"{icon} [{name_escaped}]({link})" if link else f"{icon} {name_escaped}"
            reply_markup = None
            if item['mimeType'] == 'text/csv': # تحقق من نوع الملف CSV
                keyboard = [[InlineKeyboardButton(f"📄 اقرأ هذا الـ CSV", callback_data=f'read_csv_{item["id"]}')]]
                reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(file_info_line, parse_mode='MarkdownV2', reply_markup=reply_markup)
    except Exception as e:
        logger.error(f'Error listing GDrive files: {e}')
        await update.message.reply_text(f'خطأ في عرض ملفات GDrive: {type(e).__name__}')

# --- !!! دالة button_callback الوظيفية (ليست التشخيصية) !!! ---
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """يعالج الضغط على الأزرار المضمنة للوظائف المختلفة."""
    query = update.callback_query
    # نرد على تيليجرام فورًا لتجنب انتهاء مهلة الـ callback
    await query.answer() 

    user_id = query.from_user.id
    callback_data = query.data
    
    logger.info(f"--- BUTTON CALLBACK (Functional) --- User ID: {user_id}, Callback Data: '{callback_data}'")
    
    chat_id_to_reply = query.message.chat_id if query.message else user_id

    if callback_data.startswith('read_csv_'):
        file_id = callback_data.split('_', 2)[2]
        logger.info(f"User {user_id} requested to read CSV file with ID: {file_id}")
        
        try:
            await query.edit_message_text(text=f"{query.message.text if query.message else ''}\n\n---\nجاري قراءة ملف CSV (ID: {file_id})...")
        except Exception as e:
            logger.warning(f"Could not edit original button message for read_csv_ action: {e}")
            await context.bot.send_message(chat_id=chat_id_to_reply, text=f"جاري قراءة ملف CSV (ID: {file_id})...")
        
        await context.bot.send_chat_action(chat_id=chat_id_to_reply, action=ChatAction.TYPING)

        service = await get_gdrive_service_async()
        if not service:
            await context.bot.send_message(chat_id=chat_id_to_reply, text="فشل الاتصال بـ Google Drive. لا يمكن قراءة الملف.")
            return
        
        # --- دالة داخلية لتنزيل ومعالجة CSV بشكل متزامن ---
        def download_and_process_csv_sync(gdrive_service, gdrive_file_id):
            try:
                logger.info(f"Attempting to download file with ID: {gdrive_file_id}")
                request = gdrive_service.files().get_media(fileId=gdrive_file_id)
                csv_content_stream = io.BytesIO()
                csv_content_stream.write(request.execute())
                csv_content_stream.seek(0)
                logger.info(f"Successfully downloaded content for file ID: {gdrive_file_id}")
                
                df = pd.read_csv(csv_content_stream)
                if df.empty:
                    return "ملف CSV فارغ أو لا يحتوي على بيانات يمكن قراءتها."
                
                num_rows, num_cols = df.shape
                columns = ", ".join(df.columns.tolist())
                
                # استخدام backticks ثلاثية لتنسيق الكود بشكل أفضل في تيليجرام
                output_message_unescaped = (
                    f"تمت قراءة ملف CSV (ID: {gdrive_file_id}) بنجاح!\n"
                    f"عدد الصفوف: {num_rows}\n"
                    f"عدد الأعمدة: {num_cols}\n"
                    f"أسماء الأعمدة: `{columns}`\n\n"
                    f"أول 3 صفوف من البيانات:\n"
                    f"```\n{df.head(3).to_string(index=True)}\n```"
                )
                # لا حاجة لدالة تطهير منفصلة هنا، سنستخدم parse_mode='MarkdownV2' بحذر
                # وسنهتم بتطهير الأعمدة إذا كانت تحتوي على أحرف خاصة لاحقًا إذا لزم الأمر
                return output_message_unescaped

            except pd.errors.EmptyDataError:
                logger.warning("Attempted to read an empty CSV file or stream (EmptyDataError).")
                return "ملف CSV فارغ أو لا يحتوي على بيانات."
            except HttpError as http_err: # خطأ من Google API أثناء التنزيل
                logger.error(f"HttpError downloading/processing CSV {gdrive_file_id}: {http_err}")
                return f"خطأ في تنزيل ملف CSV (ID: {gdrive_file_id}): {http_err.resp.status}"
            except Exception as e:
                logger.error(f"Error processing CSV data with pandas for {gdrive_file_id}: {e}")
                return f"حدث خطأ أثناء معالجة ملف CSV (ID: {gdrive_file_id}): {type(e).__name__}"
        # --- نهاية الدالة الداخلية ---

        analysis_result = await asyncio.to_thread(download_and_process_csv_sync, service, file_id)
        
        # إرسال نتيجة التحليل كرسالة جديدة
        # تأكد من تطهير النص إذا كان سيحتوي على أحرف Markdown خاصة
        def escape_markdown_v2(text: str) -> str:
            escape_chars = r'_*[]()~`>#+-.!{}='
            return "".join(f'\\{char}' if char in escape_chars else char for char in text)

        await context.bot.send_message(chat_id=chat_id_to_reply, text=escape_markdown_v2(analysis_result), parse_mode='MarkdownV2')

    elif callback_data == 'feedback_useful':
        try:
            await query.edit_message_text(text=f"{query.message.text if query.message else 'تم التقييم'}\n\n---\nتقييمك: إجابة مفيدة 👍")
        except Exception as e: # قد تفشل إذا كانت الرسالة الأصلية لا يمكن تعديلها أو تم تعديلها بالفعل
            logger.warning(f"Could not edit message for useful feedback: {e}")
            await context.bot.send_message(chat_id=chat_id_to_reply, text="شكرًا لتقييمك الإيجابي! 👍")
            
    elif callback_data == 'feedback_not_useful':
        try:
            await query.edit_message_text(text=f"{query.message.text if query.message else 'تم التقييم'}\n\n---\nتقييمك: إجابة غير مفيدة 👎")
        except Exception as e:
            logger.warning(f"Could not edit message for not useful feedback: {e}")
            await context.bot.send_message(chat_id=chat_id_to_reply, text="شكرًا لتقييمك. سنأخذه في الاعتبار. 👎")
    else:
        logger.warning(f"Received unknown callback_data: '{callback_data}'")
        await context.bot.send_message(chat_id=chat_id_to_reply, text=f"تم استلام ضغطة زر غير معروفة بالبيانات: {callback_data}")


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="عذراً، لم أفهم هذا الأمر.")

# --- الدالة الرئيسية لتشغيل البوت ---
if __name__ == '__main__':
    logger.info("Starting bot with GDrive (CSV read), QR, and OpenAI features...")

    if not BOT_TOKEN:
        logger.critical("CRITICAL: BOT_TOKEN is not set. Bot cannot start.")
        exit()

    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # إضافة المعالجات
    application.add_handler(CommandHandler('start', start_command))
    application.add_handler(CommandHandler('admin_test', admin_test_command))
    application.add_handler(CommandHandler('qr', qr_command_handler))
    application.add_handler(CommandHandler('testai', testai_command))
    application.add_handler(CommandHandler('gdrivefiles', list_gdrive_files_command))

    # --- !!! تسجيل معالج CallbackQueryHandler الوظيفي !!! ---
    # هذا المعالج سيلتقط كل ضغطات الأزرار، والمنطق الداخلي سيوجهها
    application.add_handler(CallbackQueryHandler(button_callback))
    # إذا أردت معالجات أكثر تحديدًا لاحقًا، يمكنك استخدام "pattern"
    # application.add_handler(CallbackQueryHandler(handle_read_csv_button, pattern=r'^read_csv_'))
    # application.add_handler(CallbackQueryHandler(handle_feedback_button, pattern=r'^feedback_'))

    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo_message))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    logger.info("Bot is now polling for updates...")
    application.run_polling()

    logger.info("Bot has stopped.")