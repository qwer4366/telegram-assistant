# main.py
import logging
import io
import json # <-- لاستيراد json
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
from googleapiclient.http import MediaIoBaseUpload # <--- لإدارة تحميل الملفات

# --- إعدادات البوت الأساسية ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ADMIN_ID_1 = 1263152179
ADMIN_ID_2 = 697852646
ADMIN_IDS = [ADMIN_ID_1, ADMIN_ID_2]

# --- إعدادات Google Drive API ---
# IMPORTANT: If you change SCOPES, delete token.json to force re-authentication.
SCOPES = ['https://www.googleapis.com/auth/drive']
UPLOAD_FOLDER_NAME = "Bot Uploads"
ALLOWED_MIME_TYPES = {
    'text/csv': 'csv',
    'application/vnd.ms-excel': 'xls',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    'application/json': 'json',
    'text/plain': 'txt'
}
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

async def get_or_create_folder_id_async(service, folder_name: str) -> str | None:
    """Searches for a folder by name. If not found, creates it. Returns folder ID or None."""
    
    # Helper function to run synchronous GDrive calls in a thread
    def _search_folder():
        try:
            logger.info(f"Searching for folder: '{folder_name}'")
            response = service.files().list(
                q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false",
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            folders = response.get('files', [])
            if folders:
                folder_id = folders[0].get('id')
                logger.info(f"Folder '{folder_name}' found with ID: {folder_id}")
                return folder_id
            return None
        except HttpError as error:
            logger.error(f"HttpError while searching for folder '{folder_name}': {error}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while searching for folder '{folder_name}': {e}")
            return None

    def _create_folder():
        try:
            logger.info(f"Folder '{folder_name}' not found. Creating it...")
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            folder_id = folder.get('id')
            logger.info(f"Folder '{folder_name}' created with ID: {folder_id}")
            return folder_id
        except HttpError as error:
            logger.error(f"HttpError while creating folder '{folder_name}': {error}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while creating folder '{folder_name}': {e}")
            return None

    folder_id = await asyncio.to_thread(_search_folder)
    if not folder_id:
        folder_id = await asyncio.to_thread(_create_folder)
    
    return folder_id

# --- معالج تحميل الملفات ---
async def upload_document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles document uploads to Google Drive."""
    if not update.message or not update.message.document:
        logger.warning("upload_document_handler called without a document.")
        # This should ideally not happen if handler is registered for filters.Document.ALL
        await update.message.reply_text("يرجى إرسال ملف ليتم تحميله.")
        return

    doc = update.message.document
    file_name = doc.file_name
    mime_type = doc.mime_type

    if mime_type not in ALLOWED_MIME_TYPES:
        logger.info(f"User {update.effective_user.id} tried to upload unsupported file: {file_name} (MIME: {mime_type})")
        allowed_ext_str = ", ".join(ALLOWED_MIME_TYPES.values())
        await update.message.reply_text(
            f"عذرًا، نوع الملف '{mime_type}' غير مدعوم.\n"
            f"الأنواع المدعومة حاليًا هي: {allowed_ext_str}."
        )
        return

    logger.info(f"User {update.effective_user.id} initiated upload for: {file_name} (MIME: {mime_type})")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_DOCUMENT)
    
    # رسالة أولية للمستخدم
    status_message = await update.message.reply_text(f"جاري معالجة ملفك '{file_name}'...")

    try:
        # 1. تنزيل الملف من تيليجرام
        logger.debug(f"Downloading '{file_name}' from Telegram...")
        tg_file = await context.bot.get_file(doc.file_id)
        file_content_bytearray = await tg_file.download_as_bytearray()
        file_content_stream = io.BytesIO(file_content_bytearray)
        logger.info(f"Successfully downloaded '{file_name}' from Telegram. Size: {len(file_content_bytearray)} bytes.")

        # 2. الاتصال بخدمة Google Drive
        await status_message.edit_text(f"جاري الاتصال بـ Google Drive لتحميل '{file_name}'...")
        service = await get_gdrive_service_async()
        if not service:
            logger.error("Failed to get Google Drive service for upload.")
            await status_message.edit_text("فشل الاتصال بخدمة Google Drive. لا يمكن تحميل الملف.")
            return

        # 3. الحصول على أو إنشاء مجلد الرفع
        await status_message.edit_text(f"جاري التحقق من مجلد '{UPLOAD_FOLDER_NAME}' في Google Drive...")
        folder_id = await get_or_create_folder_id_async(service, UPLOAD_FOLDER_NAME)
        if not folder_id:
            logger.error(f"Failed to get or create folder '{UPLOAD_FOLDER_NAME}'.")
            await status_message.edit_text(f"فشل في الوصول إلى أو إنشاء مجلد '{UPLOAD_FOLDER_NAME}' في Google Drive.")
            return
        logger.info(f"Target folder ID for '{UPLOAD_FOLDER_NAME}' is {folder_id}.")

        # 4. إعداد بيانات الملف والرفع
        await status_message.edit_text(f"جاري رفع '{file_name}' إلى Google Drive...")
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        # استخدم MIME type الأصلي للملف عند الرفع
        media = MediaIoBaseUpload(file_content_stream, mimetype=mime_type, resumable=True)
        
        # دالة الرفع المتزامنة
        def _upload_file_sync():
            try:
                logger.info(f"Starting synchronous upload of '{file_name}' to folder ID '{folder_id}'.")
                uploaded_file = service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id, name, webViewLink' 
                ).execute()
                logger.info(f"Successfully uploaded '{file_name}' (ID: {uploaded_file.get('id')}) to Google Drive.")
                return uploaded_file
            except HttpError as error:
                logger.error(f"HttpError during GDrive upload of '{file_name}': {error.resp.status} - {error._get_reason()}")
                return {'error': f"خطأ {error.resp.status} من Google Drive: {error._get_reason()}"}
            except Exception as e:
                logger.error(f"Unexpected error during GDrive upload of '{file_name}': {e}")
                return {'error': f"خطأ غير متوقع أثناء الرفع: {type(e).__name__}"}

        # تنفيذ الرفع في thread منفصل
        upload_result = await asyncio.to_thread(_upload_file_sync)

        if 'error' in upload_result:
            error_msg = upload_result['error']
            await status_message.edit_text(f"فشل تحميل الملف '{file_name}':\n{error_msg}")
        elif upload_result and upload_result.get('webViewLink'):
            success_message = (
                f"تم تحميل ملفك '{upload_result.get('name')}' بنجاح إلى Google Drive!\n"
                f"يمكنك عرضه هنا: {upload_result.get('webViewLink')}"
            )
            await status_message.edit_text(success_message)
        else:
            logger.error(f"Upload of '{file_name}' completed but no webViewLink or other error info received. Result: {upload_result}")
            await status_message.edit_text(f"اكتمل تحميل '{file_name}' ولكن حدث خطأ غير معروف في استرداد معلومات الملف.")

    except Exception as e:
        logger.error(f"An unexpected error occurred in upload_document_handler for '{file_name}': {e}", exc_info=True)
        await status_message.edit_text(f"حدث خطأ غير متوقع أثناء معالجة ملفك '{file_name}': {type(e).__name__}.")


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

# --- Helper function to escape MarkdownV2 ---
def escape_markdown_v2(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-.!{}='
    return "".join(f'\\{char}' if char in escape_chars else char for char in text)

# --- Google Drive Helper Function to display folder contents ---
async def _get_and_display_gdrive_folder_contents(
    context: ContextTypes.DEFAULT_TYPE, 
    chat_id: int, 
    folder_id: str = 'root', 
    page_token: str | None = None,
    message_id_to_edit: int | None = None
):
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    service = await get_gdrive_service_async()
    if not service:
        error_text = "لم أتمكن من الاتصال بـ Google Drive."
        if message_id_to_edit:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=message_id_to_edit, text=error_text)
        else:
            await context.bot.send_message(chat_id=chat_id, text=error_text)
        return

    current_folder_name = "Google Drive Root"
    parent_folder_id = None

    try:
        if folder_id != 'root':
            def get_folder_details_sync():
                return service.files().get(fileId=folder_id, fields='id, name, parents').execute()
            current_folder_obj = await asyncio.to_thread(get_folder_details_sync)
            current_folder_name = current_folder_obj.get('name', 'مجلد غير معروف')
            if current_folder_obj.get('parents'):
                parent_folder_id = current_folder_obj['parents'][0] # Get first parent

        logger.info(f"Listing GDrive folder: '{current_folder_name}' (ID: {folder_id}), PageToken: {page_token}")
        
        def list_files_sync():
            return service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                pageSize=10, # Keep page size manageable for Telegram messages
                fields="nextPageToken, files(id, name, mimeType, webViewLink)",
                orderBy="folder, name", # Sort by folder first, then by name
                pageToken=page_token
            ).execute()

        results = await asyncio.to_thread(list_files_sync)
        items = results.get('files', [])
        next_page_token = results.get('nextPageToken')
        
        message_text = f"📁 {escape_markdown_v2(current_folder_name)}\n\n"
        keyboard_buttons = []

        if not items and not parent_folder_id and not next_page_token:
            message_text += "المجلد فارغ."
        elif not items:
             message_text += "لا توجد عناصر أخرى هنا."


        for item in items:
            icon = "📁" if item['mimeType'] == 'application/vnd.google-apps.folder' else "📄"
            name_escaped = escape_markdown_v2(item['name'])
            file_info_line = f"{icon} {name_escaped}"
            
            item_buttons = []
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                item_buttons.append(InlineKeyboardButton(f"➡️ فتح", callback_data=f'gdrive_navigate_folder_{item["id"]}_None')) # Page token initially None for new folder
            else: # It's a file
                link = item.get('webViewLink', '')
                if link: # Add a view button if there's a direct link
                    item_buttons.append(InlineKeyboardButton("🔗 عرض", url=link))
                
                if item['mimeType'] == 'text/csv':
                    item_buttons.append(InlineKeyboardButton(f"📄 اقرأ CSV", callback_data=f'read_csv_{item["id"]}'))
                elif item['mimeType'] == 'application/json':
                     item_buttons.append(InlineKeyboardButton(f"📄 اقرأ JSON", callback_data=f'read_json_{item["id"]}'))
                elif item['mimeType'] == 'text/plain':
                     item_buttons.append(InlineKeyboardButton(f"📄 اقرأ TXT", callback_data=f'read_txt_{item["id"]}'))
            
            keyboard_buttons.append([InlineKeyboardButton(file_info_line, callback_data=f'gdrive_noop_{item["id"]}')]) # Non-interactive button for file info
            if item_buttons:
                 keyboard_buttons.append(item_buttons)


        navigation_row = []
        if parent_folder_id:
            navigation_row.append(InlineKeyboardButton("⬆️ مجلد أب", callback_data=f'gdrive_navigate_folder_{parent_folder_id}_None'))
        
        if next_page_token:
            navigation_row.append(InlineKeyboardButton("التالي ➡️", callback_data=f'gdrive_list_page_{folder_id}_{next_page_token}'))
        
        if navigation_row:
            keyboard_buttons.append(navigation_row)
        
        reply_markup = InlineKeyboardMarkup(keyboard_buttons) if keyboard_buttons else None

        if message_id_to_edit:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=message_id_to_edit, text=message_text, reply_markup=reply_markup, parse_mode='MarkdownV2')
        else:
            await context.bot.send_message(chat_id=chat_id, text=message_text, reply_markup=reply_markup, parse_mode='MarkdownV2')

    except HttpError as e:
        logger.error(f'HttpError while listing GDrive folder ID {folder_id}: {e}')
        error_text = f'خطأ في عرض محتويات المجلد (HttpError): {e.resp.status}'
        if message_id_to_edit:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=message_id_to_edit, text=error_text)
        else:
            await context.bot.send_message(chat_id=chat_id, text=error_text)
    except Exception as e:
        logger.error(f'Error listing GDrive folder ID {folder_id}: {e}', exc_info=True)
        error_text = f'خطأ غير متوقع في عرض محتويات المجلد: {type(e).__name__}'
        if message_id_to_edit:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=message_id_to_edit, text=error_text)
        else:
            await context.bot.send_message(chat_id=chat_id, text=error_text)


# --- معالج أمر Google Drive ---
async def list_gdrive_files_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    # Check for client_secret.json early, as it's crucial for any GDrive operation
    if not os.path.exists(CLIENT_SECRET_FILE):
        await update.message.reply_text(f"ملف `{CLIENT_SECRET_FILE}` غير موجود في: `{os.getcwd()}`. لا يمكن الوصول لـ Google Drive.")
        logger.error(f"Missing {CLIENT_SECRET_FILE} at {os.getcwd()} for /gdrivefiles command.")
        return

    # Send initial "Fetching..." message
    status_message = await update.message.reply_text("جاري جلب قائمة الملفات من Google Drive...")

    folder_id_to_list = 'root'
    if context.args:
        folder_id_to_list = context.args[0]
        logger.info(f"/gdrivefiles called with folder_id argument: {folder_id_to_list}")

    await _get_and_display_gdrive_folder_contents(
        context=context,
        chat_id=chat_id,
        folder_id=folder_id_to_list,
        page_token=None, # Initial call has no page token
        message_id_to_edit=status_message.message_id
    )

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
        file_id_from_cb = callback_data.split('_', 2)[2]
        logger.info(f"User {user_id} requested to read CSV file with ID: {file_id_from_cb}")
        
        original_message_text = query.message.text if query.message else "..."
        try:
            await query.edit_message_text(text=f"{original_message_text}\n\n---\nجاري قراءة ملف CSV (ID: {file_id_from_cb})...")
        except Exception as e:
            logger.warning(f"Could not edit original button message for read_csv_ action: {e}")
            # If editing fails, send a new message but don't try to edit it again later for the result
            await context.bot.send_message(chat_id=chat_id_to_reply, text=f"جاري قراءة ملف CSV (ID: {file_id_from_cb})...")
        
        await context.bot.send_chat_action(chat_id=chat_id_to_reply, action=ChatAction.TYPING)

        gdrive_service_instance = await get_gdrive_service_async()
        if not gdrive_service_instance:
            await context.bot.send_message(chat_id=chat_id_to_reply, text="فشل الاتصال بـ Google Drive. لا يمكن قراءة الملف.")
            return
        
        # --- دالة داخلية لتنزيل ومعالجة CSV بشكل متزامن ---
        def download_and_process_csv_sync(current_gdrive_service, gdrive_file_id_param):
            try:
                logger.info(f"Attempting to download file with ID: {gdrive_file_id_param}")
                request = current_gdrive_service.files().get_media(fileId=gdrive_file_id_param)
                csv_content_bytesio = io.BytesIO()
                csv_content_bytesio.write(request.execute())
                csv_content_bytesio.seek(0)
                logger.info(f"Successfully downloaded content for file ID: {gdrive_file_id_param}")
                
                df = pd.read_csv(csv_content_bytesio) # Ensure this is BytesIO
                if df.empty:
                    return "ملف CSV فارغ أو لا يحتوي على بيانات يمكن قراءتها."
                
                num_rows, num_cols = df.shape
                columns_str = ", ".join(df.columns.tolist())
                
                # Construct the message carefully, escaping only necessary parts.
                # The main data (df.head) is within a code block and doesn't need escaping.
                # Column names are put in single backticks.
                # The file ID is escaped.
                processed_gdrive_file_id = escape_markdown_v2(gdrive_file_id_param)
                processed_columns_str = escape_markdown_v2(columns_str)

                output_message = (
                    f"تمت قراءة ملف CSV (ID: {processed_gdrive_file_id}) بنجاح!\n"
                    f"عدد الصفوف: {num_rows}\n"
                    f"عدد الأعمدة: {num_cols}\n"
                    f"أسماء الأعمدة: `{processed_columns_str}`\n\n"
                    f"أول 3 صفوف من البيانات:\n"
                    f"```\n{df.head(3).to_string(index=True)}\n```"
                )
                return output_message

            except pd.errors.EmptyDataError:
                logger.warning(f"Attempted to read an empty CSV file or stream (EmptyDataError) for ID: {gdrive_file_id_param}.")
                return f"ملف CSV (ID: {escape_markdown_v2(gdrive_file_id_param)}) فارغ أو لا يحتوي على بيانات."
            except HttpError as http_err_csv: 
                logger.error(f"HttpError downloading/processing CSV {gdrive_file_id_param}: {http_err_csv}")
                return f"خطأ في تنزيل/معالجة ملف CSV (ID: {escape_markdown_v2(gdrive_file_id_param)}): {http_err_csv.resp.status}"
            except Exception as e_csv:
                logger.error(f"Error processing CSV data with pandas for {gdrive_file_id_param}: {e_csv}", exc_info=True)
                return f"حدث خطأ أثناء معالجة ملف CSV (ID: {escape_markdown_v2(gdrive_file_id_param)}): {type(e_csv).__name__}"
        # --- نهاية الدالة الداخلية ---

        analysis_result_csv = await asyncio.to_thread(download_and_process_csv_sync, gdrive_service_instance, file_id_from_cb)
        
        # The analysis_result_csv string is already formatted and escaped appropriately by the sync function.
        await context.bot.send_message(chat_id=chat_id_to_reply, text=analysis_result_csv, parse_mode='MarkdownV2')

    elif callback_data.startswith('read_json_') or callback_data.startswith('read_txt_'):
        file_id_from_cb = callback_data.split('_', 2)[2]
        action_type = callback_data.split('_', 2)[0] # 'read_json' or 'read_txt'
        
        original_message_text = query.message.text if query.message else "..."
        processing_message_text = f"{original_message_text}\n\n---\nجاري قراءة ملف {action_type.split('_')[1].upper()} (ID: {file_id_from_cb})..."
        
        try:
            await query.edit_message_text(text=processing_message_text)
        except Exception as e_edit:
            logger.warning(f"Could not edit original button message for {action_type} action: {e_edit}")
            await context.bot.send_message(chat_id=chat_id_to_reply, text=f"جاري قراءة ملف {action_type.split('_')[1].upper()} (ID: {file_id_from_cb})...")

        await context.bot.send_chat_action(chat_id=chat_id_to_reply, action=ChatAction.TYPING)
        gdrive_service_instance = await get_gdrive_service_async()

        if not gdrive_service_instance:
            await context.bot.send_message(chat_id=chat_id_to_reply, text="فشل الاتصال بـ Google Drive. لا يمكن قراءة الملف.")
            return

        def download_file_content_sync(current_gdrive_service, gdrive_file_id_param):
            try:
                logger.info(f"Attempting to download file content for ID: {gdrive_file_id_param} for {action_type}")
                request = current_gdrive_service.files().get_media(fileId=gdrive_file_id_param)
                content_bytesio = io.BytesIO()
                content_bytesio.write(request.execute())
                content_bytesio.seek(0)
                logger.info(f"Successfully downloaded content for file ID: {gdrive_file_id_param}")
                return content_bytesio.getvalue() # Return bytes
            except HttpError as http_err_download:
                logger.error(f"HttpError downloading file {gdrive_file_id_param}: {http_err_download}")
                return f"خطأ في تنزيل الملف (ID: {gdrive_file_id_param}): {http_err_download.resp.status}"
            except Exception as e_download:
                logger.error(f"Error downloading file {gdrive_file_id_param}: {e_download}", exc_info=True)
                return f"حدث خطأ أثناء تنزيل الملف (ID: {gdrive_file_id_param}): {type(e_download).__name__}"

        file_bytes = await asyncio.to_thread(download_file_content_sync, gdrive_service_instance, file_id_from_cb)

        if isinstance(file_bytes, str): # Error message from download
            await context.bot.send_message(chat_id=chat_id_to_reply, text=escape_markdown_v2(file_bytes))
            return

        processed_content_message = f"خطأ غير معروف في معالجة الملف {action_type.split('_')[1].upper()}."

        if action_type == 'read_json':
            try:
                json_string = file_bytes.decode('utf-8')
                data = json.loads(json_string)
                
                # Attempt to normalize into a DataFrame
                try:
                    df = pd.json_normalize(data)
                    num_rows, num_cols = df.shape
                    info_buffer = io.StringIO()
                    df.info(buf=info_buffer)
                    info_str = info_buffer.getvalue()
                    
                    processed_content_message = (
                        f"تمت قراءة ملف JSON (ID: {escape_markdown_v2(file_id_from_cb)}) كجدول:\n"
                        f"```\n{info_str}\n```\n" # info_str is already formatted text, keep in simple code block
                        f"أول 3 صفوف:\n"
                        f"```\n{df.head(3).to_string(index=True)}\n```"
                    )
                except Exception as e_df: 
                    logger.warning(f"Could not normalize JSON from {file_id_from_cb} to DataFrame ({e_df}), pretty-printing instead.")
                    pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
                    max_chars_json = 3500
                    truncation_note_json = ""
                    if len(pretty_json) > max_chars_json: 
                        pretty_json = pretty_json[:max_chars_json]
                        truncation_note_json = "\n... (مقتطع لكونه طويل جداً)"
                    processed_content_message = (
                        f"محتوى ملف JSON (ID: {escape_markdown_v2(file_id_from_cb)}):\n"
                        f"```json\n{pretty_json}{truncation_note_json}\n```"
                    )
            except json.JSONDecodeError:
                logger.error(f"JSONDecodeError for file ID: {file_id_from_cb}")
                processed_content_message = f"خطأ: ملف JSON (ID: {escape_markdown_v2(file_id_from_cb)}) غير صالح أو لا يمكن تحليله."
            except UnicodeDecodeError:
                logger.error(f"UnicodeDecodeError for JSON file ID: {file_id_from_cb}")
                processed_content_message = f"خطأ: لا يمكن فك ترميز ملف JSON (ID: {escape_markdown_v2(file_id_from_cb)}). تأكد أنه UTF-8."
            except Exception as e_json_proc:
                logger.error(f"Error processing JSON file {file_id_from_cb}: {e_json_proc}", exc_info=True)
                processed_content_message = f"خطأ غير متوقع في معالجة JSON (ID: {escape_markdown_v2(file_id_from_cb)}): {type(e_json_proc).__name__}"
        
        elif action_type == 'read_txt':
            try:
                text_content = file_bytes.decode('utf-8')
                max_chars_txt = 3500
                max_lines_txt = 50
                display_content = ""
                truncation_note_txt = ""

                if not text_content.strip():
                    display_content = "(الملف النصي فارغ)"
                else:
                    lines = text_content.splitlines()
                    if len(lines) > max_lines_txt:
                        display_content = "\n".join(lines[:max_lines_txt])
                        truncation_note_txt = f"\n... (مقتطع، تم عرض أول {max_lines_txt} سطراً)"
                        if len(display_content) > max_chars_txt: # Check char limit on the line-truncated content
                            display_content = display_content[:max_chars_txt]
                            # Update truncation note if both apply, prioritize line limit message
                            truncation_note_txt = f"\n... (مقتطع، تم عرض أول {max_lines_txt} سطراً وتجاوز حد الحروف)"
                    elif len(text_content) > max_chars_txt:
                        display_content = text_content[:max_chars_txt]
                        truncation_note_txt = f"\n... (مقتطع، تجاوز الحد الأقصى للحروف وهو {max_chars_txt})"
                    else:
                        display_content = text_content
                
                processed_content_message = (
                    f"محتوى ملف TXT (ID: {escape_markdown_v2(file_id_from_cb)}):\n"
                    f"```text\n{display_content}{truncation_note_txt}\n```"
                )
            except UnicodeDecodeError:
                logger.error(f"UnicodeDecodeError for TXT file ID: {file_id_from_cb}")
                processed_content_message = f"خطأ: لا يمكن فك ترميز الملف النصي (ID: {escape_markdown_v2(file_id_from_cb)}). تأكد أنه UTF-8."
            except Exception as e_txt_proc:
                logger.error(f"Error processing TXT file {file_id_from_cb}: {e_txt_proc}", exc_info=True)
                processed_content_message = f"خطأ غير متوقع في معالجة TXT (ID: {escape_markdown_v2(file_id_from_cb)}): {type(e_txt_proc).__name__}"
        
        # The processed_content_message is constructed with MarkdownV2 in mind (code blocks for data).
        # No further blanket escaping is needed here.
        await context.bot.send_message(chat_id=chat_id_to_reply, text=processed_content_message, parse_mode='MarkdownV2')

    elif callback_data.startswith('gdrive_navigate_folder_') or callback_data.startswith('gdrive_list_page_'):
        parts = callback_data.split('_')
        action_type = parts[1] # 'navigate' or 'list' # This was parts[1] before, should be parts[1] as well.
        target_folder_id = parts[3]
        page_token_from_cb = parts[4] if len(parts) > 4 else None
        if page_token_from_cb == 'None': page_token_from_cb = None # Handle string "None"
        
        logger.info(f"GDrive navigation/pagination: Action: {action_type}, Target Folder: {target_folder_id}, Page Token: {page_token_from_cb}")

        await _get_and_display_gdrive_folder_contents(
            context=context,
            chat_id=chat_id_to_reply,
            folder_id=target_folder_id,
            page_token=page_token_from_cb,
            message_id_to_edit=query.message.message_id if query.message else None
        )
    
    elif callback_data.startswith('gdrive_noop_'):
        logger.debug(f"GDrive no-op action for data: {callback_data}. Acknowledging.")
        # No action needed, but answering the query prevents the "loading" icon on the button.
        # await query.answer() # Already answered at the beginning of button_callback

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
    
    # --- معالج تحميل الملفات ---
    application.add_handler(MessageHandler(filters.Document.ALL, upload_document_handler))

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