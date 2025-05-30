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

import fitz  # PyMuPDF
import pytesseract
from io import BytesIO # Already have import io, but BytesIO is useful directly
from PIL import Image

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
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
from googleapiclient.http import MediaIoBaseUpload

# --- إعدادات البوت الأساسية ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DRIVE_BOT_UPLOAD_FOLDER_ID = os.getenv("DRIVE_BOT_UPLOAD_FOLDER_ID")
if not DRIVE_BOT_UPLOAD_FOLDER_ID:
    logger.warning("!!! DRIVE_BOT_UPLOAD_FOLDER_ID is not set in .env. File uploads to Drive will fail. Please create a folder in Google Drive and set its ID as DRIVE_BOT_UPLOAD_FOLDER_ID in the .env file.")

ADMIN_ID_1 = 1263152179
ADMIN_ID_2 = 697852646
ADMIN_IDS = [ADMIN_ID_1, ADMIN_ID_2]

# --- إعدادات Google Drive API ---
# Full access scope for Google Drive (read, write, create, delete).
# If you previously authenticated with read-only, you might need to delete token.json and re-authenticate.
SCOPES = ['https://www.googleapis.com/auth/drive']
CLIENT_SECRET_FILE = 'client_secret.json'
TOKEN_FILE = 'token.json'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO # يمكنك إعادته إلى INFO الآن أو تركه DEBUG إذا أردت المزيد من التفاصيل
)
logger = logging.getLogger(__name__)
# logging.getLogger("httpx").setLevel(logging.WARNING) # يمكنك إعادة تفعيل هذا لتقليل ضوضاء httpx
# logging.getLogger("telegram.ext").setLevel(logging.DEBUG) # يمكنك ترك هذا DEBUG أو إعادته لـ INFO


# --- Funciones de extracción de texto PDF ---
def sync_extract_text_from_pdf(pdf_bytes: bytes) -> tuple[int, str]:
    text = ""
    num_pages = 0
    doc = None  # Initialize doc to None for the finally block
    pdf_stream = BytesIO(pdf_bytes) # Keep stream open for potential OCR pass

    try:
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        num_pages = doc.page_count
        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            text += page.get_text("text") # Specify text format

        if num_pages > 0 and (not text.strip() or len(text.strip()) < 50): # Condition for OCR
            logger.info(f"Initial text extraction short or empty (length {len(text.strip())}, pages {num_pages}). Attempting OCR.")
            ocr_full_text = ""
            # Re-open doc if it was closed or use a fresh stream; for now, assume doc is fine.
            # If doc was closed, we'd need: pdf_stream.seek(0); doc = fitz.open(stream=pdf_stream, filetype="pdf")
            # However, fitz.open already read from the stream, so for OCR, we might need to reopen from original bytes if stream is consumed.
            # For simplicity, let's assume the 'doc' object can still be used if it wasn't closed.
            # If PyMuPDF consumes the stream, it's safer to re-create BytesIO for each pass or keep original bytes.
            # For now, we'll assume 'doc' is reusable for page loading if not closed.

            # Re-iterate for OCR if doc is still valid for page loading
            # If `doc.load_page` fails after initial text extraction, `doc` would need to be reopened.
            # This example assumes `doc` can still be used. A more robust solution might re-open from `pdf_bytes`.
            
            # Ensure doc is re-opened if it was closed or stream was fully consumed.
            # For this implementation, we'll assume `doc` is from the initial try block and still usable.
            # If `page.get_text()` fully consumes parts of the stream needed by `get_pixmap`,
            # then `doc` should be reopened using a fresh BytesIO stream from `pdf_bytes`.
            # Let's try to use the existing 'doc' and pages.
            
            # If initial text extraction already iterated through pages, the `doc` object is fine.

            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                try:
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    img = Image.open(BytesIO(img_bytes))
                    # Specify languages: Arabic ('ara') and English ('eng')
                    ocr_text_page = pytesseract.image_to_string(img, lang='ara+eng')
                    ocr_full_text += ocr_text_page + "\n" # Add newline between pages
                except pytesseract.TesseractNotFoundError:
                    logger.error("Tesseract is not installed or not found in PATH.")
                    # Return original short text or error if Tesseract is missing
                    return num_pages, text.strip() if text.strip() else "OCR Error: Tesseract not found."
                except Exception as ocr_exc:
                    logger.error(f"Error during OCR for page {page_num + 1}: {ocr_exc}")
                    # Continue to next page or return what we have / original text
            
            if ocr_full_text.strip():
                logger.info("OCR process yielded text. Using OCR text.")
                text = ocr_full_text
            else:
                logger.info("OCR process did not yield any new text.")

        return num_pages, text.strip()

    except fitz.errors.FitzError as e:
        logger.error(f"PyMuPDF (Fitz) error processing PDF: {e}")
        # Consider if OCR should be attempted even if PyMuPDF fails initially.
        # For now, returning error. A more advanced flow could try OCR directly on images if fitz fails.
        return 0, "Error processing PDF with PyMuPDF."
    except Exception as e:
        logger.error(f"General error in sync_extract_text_from_pdf: {e}")
        return 0, f"General error processing PDF: {type(e).__name__}"
    finally:
        if doc:
            doc.close()
        # pdf_stream.close() # Close the BytesIO stream

async def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[int, str]:
    # Runs the synchronous PDF extraction in a separate thread
    return await asyncio.to_thread(sync_extract_text_from_pdf, pdf_bytes)


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

# --- Funciones de subida a Google Drive ---
def upload_to_gdrive_sync(gdrive_service, file_bytes: bytes, file_name: str, mime_type: str, folder_id: str) -> str | None:
    try:
        media_body = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mime_type, resumable=True)
        file_metadata = {'name': file_name}
        if folder_id and folder_id.strip() != "": # Ensure folder_id is not empty or just whitespace
            file_metadata['parents'] = [folder_id]
        
        created_file = gdrive_service.files().create(
            body=file_metadata,
            media_body=media_body,
            fields='id, webViewLink'
        ).execute()
        
        file_link = created_file.get('webViewLink')
        logger.info(f"Uploaded file '{file_name}' to Drive. ID: {created_file.get('id')}, Link: {file_link}")
        return file_link
    except HttpError as error:
        logger.error(f"HttpError during GDrive upload for '{file_name}': {error}")
        return None
    except Exception as e:
        logger.error(f"General error during GDrive upload for '{file_name}': {e}")
        return None

async def upload_to_gdrive_async(gdrive_service, file_bytes: bytes, file_name: str, mime_type: str, folder_id: str) -> str | None:
    return await asyncio.to_thread(upload_to_gdrive_sync, gdrive_service, file_bytes, file_name, mime_type, folder_id)

# --- (بقية الدوال مثل start_command, echo_message, admin_test_command, generate_qr_image, qr_command_handler, get_openai_response, testai_command كما هي) ---

# --- PDF Document Handler ---
async def handle_pdf_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Received a PDF file from user {update.effective_user.id} - Name: {update.message.document.file_name}")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    # Inform the user that processing has started
    processing_message = await update.message.reply_text("جاري معالجة ملف PDF، يرجى الانتظار...")

    file_id = update.message.document.file_id
    try:
        pdf_file = await context.bot.get_file(file_id)
        # Using download_as_bytearray first, then converting to bytes
        pdf_bytearray = await pdf_file.download_as_bytearray()
        pdf_bytes = bytes(pdf_bytearray) 

        num_pages, extracted_text = await extract_text_from_pdf(pdf_bytes)

        if not extracted_text or extracted_text == "Error processing PDF.":
            reply_text = "عذراً، لم أتمكن من استخلاص النص من ملف PDF هذا. قد يكون الملف فارغًا، تالفًا، أو محميًا بكلمة مرور، أو عبارة عن صورة ممسوحة ضوئيًا (تحتاج OCR لم يتم تنفيذه بعد)."
            logger.warning(f"PDF processing failed or returned empty/error text for file_id: {file_id}")
        else:
            # Escape for MarkdownV2 - This basic escape might need to be more robust
            # It's generally safer to avoid Markdown if the text can be complex, or use HTML parse mode
            def escape_md_v2(text: str) -> str:
                # Basic set of characters to escape for MarkdownV2
                escape_chars = r'_*[]()~`>#+-.!{}='
                return "".join(f'\\{char}' if char in escape_chars else char for char in text)

            text_preview_escaped = escape_md_v2(extracted_text[:200])
            
            reply_text = (
                f"تمت معالجة ملف PDF بنجاح\\.\n"
                f"عدد الصفحات: {num_pages}\n\n"
                f"أول 200 حرف من النص المستخلص:\n"
                f"{text_preview_escaped}"
            )
            logger.info(f"Successfully extracted text from PDF file_id: {file_id}. Pages: {num_pages}. Preview: {extracted_text[:50]}...")
        
        # Edit the "Processing..." message with the result
        await context.bot.edit_message_text(chat_id=processing_message.chat_id, message_id=processing_message.message_id, text=reply_text, parse_mode='MarkdownV2')

    except Exception as e:
        logger.error(f"Error in handle_pdf_document for file_id {file_id}: {type(e).__name__} - {e}")
        # Edit the "Processing..." message with an error
        await context.bot.edit_message_text(chat_id=processing_message.chat_id, message_id=processing_message.message_id, text="حدث خطأ غير متوقع أثناء معالجة ملف PDF.")

# --- Utility function to escape text for MarkdownV2 ---
def escape_markdown_v2(text: str) -> str:
    """Escapes special characters for Telegram MarkdownV2."""
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)
    escape_chars = r'_*[]()~`>#+-.!{}='
    return "".join(f'\\{char}' if char in escape_chars else char for char in text)

# --- Generic Document Handler for Uploading to GDrive ---
async def handle_document_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not DRIVE_BOT_UPLOAD_FOLDER_ID or DRIVE_BOT_UPLOAD_FOLDER_ID.strip() == "":
        logger.error("DRIVE_BOT_UPLOAD_FOLDER_ID is not set. Cannot upload file.")
        await update.message.reply_text("عذرًا، لم يتم تعيين مجلد الرفع في Google Drive. يرجى الاتصال بمسؤول البوت.")
        return

    doc = update.message.document
    file_id = doc.file_id
    original_file_name = doc.file_name or "untitled_document" # Fallback if file_name is None
    mime_type = doc.mime_type or "application/octet-stream"  # Default MIME type

    logger.info(f"Received document: '{original_file_name}' (MIME: {mime_type}, ID: {file_id}) for GDrive upload.")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_DOCUMENT)
    
    try:
        bot_file = await context.bot.get_file(file_id)
        file_bytes_array = await bot_file.download_as_bytearray()
        file_bytes = bytes(file_bytes_array)

        service = await get_gdrive_service_async()
        if not service:
            await update.message.reply_text("تعذر الاتصال بخدمة Google Drive. لا يمكن رفع الملف.")
            logger.error("Failed to get Google Drive service in handle_document_upload.")
            return

        upload_link = await upload_to_gdrive_async(service, file_bytes, original_file_name, mime_type, DRIVE_BOT_UPLOAD_FOLDER_ID)

        if upload_link:
            escaped_file_name = escape_markdown_v2(original_file_name)
            # The link itself is a URL, typically doesn't need escaping for the URL part of Markdown [text](url)
            # However, if the link text (which is also the filename here) has special chars, it needs escaping.
            reply_text = f"تم رفع الملف '{escaped_file_name}' بنجاح إلى Google Drive\\.\nيمكنك الوصول إليه عبر الرابط: [{escaped_file_name}]({upload_link})"
            await update.message.reply_text(reply_text, parse_mode='MarkdownV2')
        else:
            escaped_file_name = escape_markdown_v2(original_file_name)
            reply_text = f"حدث خطأ أثناء محاولة رفع الملف '{escaped_file_name}' إلى Google Drive\\."
            await update.message.reply_text(reply_text, parse_mode='MarkdownV2')
            
    except Exception as e:
        logger.error(f"Unexpected error in handle_document_upload for '{original_file_name}': {type(e).__name__} - {e}")
        escaped_file_name = escape_markdown_v2(original_file_name)
        await update.message.reply_text(f"عذرًا، حدث خطأ غير متوقع أثناء معالجة الملف '{escaped_file_name}'\\. يرجى المحاولة مرة أخرى لاحقًا\\.", parse_mode='MarkdownV2')


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome_message = (
        f"أهلاً بك يا {user.first_name}!\n"
        "أنا بوتك المساعد. يمكنك استخدام الأزرار بالأسفل للتنقل بين الوظائف الرئيسية أو كتابة الأوامر مباشرة.\n\n"
        "بعض الأوامر المتاحة:\n"
        "▫️ /qr <نص> - لإنشاء QR Code\n"
        "▫️ /testai <سؤال> - لطرح سؤال على OpenAI\n"
        # The buttons will now represent these actions primarily
        # "▫️ /gdrivefiles - لعرض ملفات Google Drive\n" 
        # "▫️ /askdata <سؤال> - لطرح سؤال عن بياناتك\n"
        "This bot is being enhanced by Jules!\n"
    )

    # Define the main reply keyboard
    main_keyboard_layout = [
        [KeyboardButton("🗂️ ملفاتي في Drive"), KeyboardButton("📄 معالجة PDF")],
        [KeyboardButton("📤 رفع ملف إلى Drive"), KeyboardButton("❓ اسأل عن بياناتي")]
        # Future buttons like "⚙️ إعدادات" or "🆘 مساعدة" can be added here
    ]
    reply_markup = ReplyKeyboardMarkup(
        main_keyboard_layout, 
        resize_keyboard=True, 
        one_time_keyboard=False # Keep the keyboard open until explicitly closed
    )

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=welcome_message,
        reply_markup=reply_markup
    )
    logger.info(f"User {user.id} ({user.first_name}) started the bot and received main keyboard.")

# --- Handlers for Reply Keyboard Buttons ---
async def prompt_pdf_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} clicked '📄 معالجة PDF' button.")
    await update.message.reply_text("الرجاء إرسال ملف PDF الذي تريد معالجته.")

async def prompt_general_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} clicked '📤 رفع ملف إلى Drive' button.")
    await update.message.reply_text("الرجاء إرسال الملف الذي تريد رفعه إلى Google Drive.")

# Note: For "🗂️ ملفاتي في Drive" and "❓ اسأل عن بياناتي",
# we will use Regex handlers to directly call the existing command functions.
# This requires those functions to be compatible.
# If they are not (e.g. they rely on context.args from a CommandHandler),
# then wrapper functions would be needed here as well.
# For now, we assume list_gdrive_files_command and askdata_command can be called.
# If askdata_command expects args, it will reply with its usage message.

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

# --- /askdata Command Handler ---
async def askdata_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("الرجاء إدخال سؤالك بعد الأمر. مثال: `/askdata ما هو متوسط المبيعات لمنتج س؟`")
        return

    user_question = " ".join(context.args)
    logger.info(f"User {update.effective_user.id} asked via /askdata: '{user_question}'")

    thinking_message = await update.message.reply_text("لحظات، أفكر في سؤالك عن البيانات... 🧠")

    if not OPENAI_API_KEY:
        await context.bot.edit_message_text(
            chat_id=thinking_message.chat_id,
            message_id=thinking_message.message_id,
            text="مفتاح OpenAI API غير مُعد. لا يمكن معالجة طلب /askdata."
        )
        logger.error("OPENAI_API_KEY is not set. Cannot process /askdata.")
        return

    # Prepare a more detailed prompt for data-related questions
    prompt_for_openai = (
        "أنت مساعد ذكاء اصطناعي متخصص في تحليل البيانات والإجابة على الأسئلة المتعلقة بها. "
        "المستخدم سيسألك عن بيانات قد تكون في ملفات CSV, Excel, أو قواعد بيانات. "
        "حاول الإجابة على سؤاله بشكل مباشر ومفيد. "
        "إذا كان السؤال يتطلب عملية حسابية معقدة أو البحث في ملف بيانات ولم يكن لديك وصول مباشر للملف، "
        "يمكنك توضيح الخطوات أو نوع الاستعلام الذي قد يحتاجه المستخدم "
        "(مثلاً، كود Pandas تقريبي أو استعلام SQL). "
        "لا تخترع بيانات إذا لم تكن متوفرة.\n\n"
        f"سؤال المستخدم: {user_question}"
    )

    response = await get_openai_response(OPENAI_API_KEY, prompt_for_openai)

    try:
        await context.bot.edit_message_text(
            chat_id=thinking_message.chat_id,
            message_id=thinking_message.message_id,
            text=response
            # TODO: Consider adding feedback buttons similar to testai_command later
        )
    except Exception as e:
        logger.error(f"Error editing 'thinking' message for /askdata: {e}")
        # Fallback to sending a new message if editing fails
        await update.message.reply_text(response)


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
    application.add_handler(CommandHandler('askdata', askdata_command)) # Added /askdata handler
    application.add_handler(CommandHandler('gdrivefiles', list_gdrive_files_command))

    # --- !!! تسجيل معالج CallbackQueryHandler الوظيفي !!! ---
    # هذا المعالج سيلتقط كل ضغطات الأزرار، والمنطق الداخلي سيوجهها
    application.add_handler(CallbackQueryHandler(button_callback))
    # إذا أردت معالجات أكثر تحديدًا لاحقًا، يمكنك استخدام "pattern"
    # application.add_handler(CallbackQueryHandler(handle_read_csv_button, pattern=r'^read_csv_'))
    # application.add_handler(CallbackQueryHandler(handle_feedback_button, pattern=r'^feedback_'))

    # --- PDF Handler ---
    # This should be placed before the generic echo_message and unknown_command handlers
    # to ensure it catches PDF files specifically.
    pdf_file_handler = MessageHandler(filters.Document.FileExtension("pdf"), handle_pdf_document)
    application.add_handler(pdf_file_handler)

    # --- Generic Document Handler (for all other document types) ---
    # This should be after specific document handlers like PDF, but before generic message handlers.
    doc_upload_handler = MessageHandler(filters.Document.ALL, handle_document_upload)
    application.add_handler(doc_upload_handler)
    
    # --- Handlers for main keyboard button presses ---
    # These should be placed before the generic echo_message handler.
    application.add_handler(MessageHandler(filters.Regex(r"^🗂️ ملفاتي في Drive$"), list_gdrive_files_command))
    application.add_handler(MessageHandler(filters.Regex(r"^❓ اسأل عن بياناتي$"), askdata_command))
    application.add_handler(MessageHandler(filters.Text(["📄 معالجة PDF"]), prompt_pdf_upload))
    application.add_handler(MessageHandler(filters.Text(["📤 رفع ملف إلى Drive"]), prompt_general_upload))

    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo_message))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command)) # Keep this last for unhandled commands
    
    logger.info("Bot is now polling for updates...")
    application.run_polling()

    logger.info("Bot has stopped.")