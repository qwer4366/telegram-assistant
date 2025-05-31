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

import json # For JSON processing
import csv # For CSV schema extraction
import re # For parsing --file in /askdata
from rapidfuzz import fuzz, process as rapidfuzz_process # For fuzzy file name matching

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
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownloader

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

# --- Conversation Handler States for /searchdata ---
SELECT_FILE, SELECT_COLUMN, SELECT_MATCH_TYPE, INPUT_SEARCH_VALUE = range(4)
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

# --- OpenAI Helper Function (Modified) ---
async def get_openai_response(api_key: str, messages: list) -> str:
    try:
        logger.info(f"Sending request to OpenAI API with messages: {messages}")
        def generate_sync():
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini", # Or your preferred model
                messages=messages,
                max_tokens=1000 # Increased max_tokens for potentially more detailed data explanations
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

# --- Funciones para Búsqueda de Archivos y Schema en Google Drive ---
def find_file_in_gdrive_sync(gdrive_service, filename_query: str) -> tuple[str | None, str | None, str | None]:
    try:
        # Fetch a broader set of files to perform fuzzy matching on
        # For now, fetching from root, ordered by most recent. This can be refined.
        # Consider targetting specific folders if context allows, or increasing pageSize if necessary.
        drive_query = "trashed = false and mimeType != 'application/vnd.google-apps.folder'" # Exclude folders from fuzzy search pool for now
        logger.info(f"Fetching up to 100 files from GDrive for fuzzy matching against: '{filename_query}'")
        results = gdrive_service.files().list(
            q=drive_query,
            pageSize=100,  # Fetch a decent pool of candidates
            orderBy="modifiedTime desc",
            fields="files(id, name, mimeType)"
        ).execute()

        items = results.get('files', [])
        if not items:
            logger.info("No files found in GDrive to perform fuzzy matching.")
            return None, None, None

        # Prepare choices for rapidfuzz: a dictionary mapping file ID to file name
        choices = {item['id']: item['name'] for item in items}

        # Use rapidfuzz to find the best match above a certain score cutoff
        # WRatio is good for finding matches even when words are out of order or strings are of different lengths
        best_match = rapidfuzz_process.extractOne(filename_query, choices, scorer=fuzz.WRatio, score_cutoff=70) # score_cutoff (0-100)

        if best_match:
            # best_match is a tuple: (matched_name_from_choices, score, original_key_from_choices (which is file_id))
            matched_name_str, score, file_id_str = best_match
            logger.info(f"Fuzzy match for '{filename_query}': Found '{matched_name_str}' (ID: {file_id_str}) with score {score:.2f}%.")

            # Retrieve the original item to get its mimeType
            original_item = next((item for item in items if item['id'] == file_id_str), None)
            if original_item:
                return original_item.get('id'), original_item.get('name'), original_item.get('mimeType')
            else:
                # This should ideally not happen if file_id_str came from our choices keys
                logger.error(f"Consistency error: Could not find original item for ID {file_id_str} after fuzzy matching.")
                return None, None, None
        else:
            logger.info(f"No suitable fuzzy match found for '{filename_query}' with cutoff 70%.")
            return None, None, None

    except HttpError as error:
        logger.error(f"HttpError during GDrive file list for fuzzy matching '{filename_query}': {error}")
        return None, None, None
    except Exception as e:
        logger.error(f"General error during GDrive file list for fuzzy matching '{filename_query}': {e}", exc_info=True)
        return None, None, None

async def find_file_in_gdrive_async(gdrive_service, filename_query: str) -> tuple[str | None, str | None, str | None]:
    return await asyncio.to_thread(find_file_in_gdrive_sync, gdrive_service, filename_query)

def get_schema_from_file_sync(gdrive_service, file_id: str, mime_type: str) -> list[str] | None: # Return type updated
    try:
        logger.info(f"Attempting to download file ID {file_id} for schema extraction (MIME: {mime_type}).")
        request = gdrive_service.files().get_media(fileId=file_id)
        file_content_stream = io.BytesIO()
        downloader = MediaIoBaseDownloader(file_content_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            # logger.debug(f"Schema download progress: {int(status.progress() * 100)}%.") # Can be verbose

        file_bytes = file_content_stream.getvalue()
        logger.info(f"File ID {file_id} downloaded successfully for schema extraction.")

        column_names = []
        if mime_type == 'text/csv':
            try:
                # Try decoding with utf-8 first, then with common alternatives if it fails
                text_content = None
                common_encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'windows-1256'] # Added windows-1256 for Arabic
                for encoding in common_encodings:
                    try:
                        text_content = file_bytes.decode(encoding)
                        logger.info(f"Decoded CSV {file_id} with {encoding}.")
                        break
                    except UnicodeDecodeError:
                        logger.debug(f"Failed to decode CSV {file_id} with {encoding}.")

                if text_content is None:
                    logger.error(f"Could not decode CSV {file_id} with any common encoding.")
                    return "تعذر فك ترميز ملف CSV باستخدام الترميزات الشائعة."

                csv_file = io.StringIO(text_content)
                reader = csv.reader(csv_file)
                column_names = next(reader) # Get header row
            except StopIteration: # Empty file
                logger.warning(f"CSV file {file_id} is empty or has no header.")
                return "ملف CSV فارغ أو لا يحتوي على رؤوس أعمدة."
            except csv.Error as e_csv:
                logger.error(f"CSV processing error for {file_id}: {e_csv}")
                return f"خطأ في معالجة ملف CSV: {e_csv}"
            except UnicodeDecodeError as e_unicode: # Fallback, though loop above should handle
                logger.error(f"Final UnicodeDecodeError for CSV {file_id}: {e_unicode}")
                return "خطأ في فك ترميز ملف CSV. تأكد من أن الملف بترميز صحيح (مثل UTF-8)."

        elif mime_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            try:
                # Reading only the header row by using nrows=0 and then getting columns
                # Forcing openpyxl for xlsx if available, as xlrd might be deprecated/removed for xlsx
                engine = 'openpyxl' if mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' else None
                df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, nrows=0, engine=engine)
                column_names = df.columns.tolist()
            except Exception as e_excel: # Catching a broader range of pandas/excel reader errors
                logger.error(f"Excel processing error for {file_id}: {e_excel}")
                return f"خطأ في معالجة ملف Excel: {e_excel}. تأكد أن الملف غير تالف."
        else:
            logger.warning(f"Unsupported mime_type '{mime_type}' for schema extraction from file ID {file_id}.")
            return None # Should not be called for unsupported types based on askdata_command logic

        if column_names:
            return column_names # Return list of column names
        else:
            # This case might occur if CSV is empty after header or Excel sheet has no columns
            logger.warning(f"No column names extracted for file ID {file_id} (MIME: {mime_type}).")
            return None # Return None if no columns found or error in specific parsing

    except HttpError as error:
        logger.error(f"HttpError during schema extraction (download) for file ID {file_id}: {error}")
        return None # Indicate error by returning None
    except Exception as e:
        logger.error(f"General error in get_schema_from_file_sync for file ID {file_id}: {e}", exc_info=True)
        return None # Indicate error by returning None

async def get_schema_from_file_async(gdrive_service, file_id: str, mime_type: str) -> list[str] | None: # Return type updated
    return await asyncio.to_thread(get_schema_from_file_sync, gdrive_service, file_id, mime_type)

# --- Funciones para Búsqueda de Datos en Archivos ---
def perform_actual_search_sync(gdrive_service, file_id: str, mime_type: str, column_name: str, match_type: str, search_value: str) -> pd.DataFrame | str:
    try:
        logger.info(f"Starting actual search: File ID {file_id}, Column '{column_name}', Match '{match_type}', Value '{search_value}'")
        # Download file content
        request = gdrive_service.files().get_media(fileId=file_id)
        file_content_stream = io.BytesIO()
        downloader = MediaIoBaseDownloader(file_content_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        file_bytes = file_content_stream.getvalue()
        logger.info(f"File {file_id} downloaded for search.")

        # Load DataFrame
        df = None
        if mime_type == 'text/csv':
            try:
                text_content = None
                common_encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'windows-1256']
                for encoding in common_encodings:
                    try:
                        text_content = file_bytes.decode(encoding)
                        logger.info(f"Decoded CSV {file_id} with {encoding} for search.")
                        break
                    except UnicodeDecodeError:
                        logger.debug(f"Failed to decode CSV {file_id} with {encoding} for search.")
                if text_content is None:
                    return "خطأ: تعذر فك ترميز ملف CSV."
                df = pd.read_csv(io.StringIO(text_content))
            except Exception as e_csv_load:
                logger.error(f"Error loading CSV into DataFrame for search (File ID: {file_id}): {e_csv_load}")
                return f"خطأ في تحميل ملف CSV: {e_csv_load}"
        elif mime_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            try:
                engine = 'openpyxl' if mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' else None
                df = pd.read_excel(io.BytesIO(file_bytes), engine=engine)
            except Exception as e_excel_load:
                logger.error(f"Error loading Excel into DataFrame for search (File ID: {file_id}): {e_excel_load}")
                return f"خطأ في تحميل ملف Excel: {e_excel_load}"
        else:
            return f"نوع الملف {mime_type} غير مدعوم للبحث."

        if df is None or df.empty:
            return "الملف فارغ أو تعذر تحميل البيانات."
        if column_name not in df.columns:
            return f"العمود '{column_name}' غير موجود في الملف."

        # Filter DataFrame
        filtered_df = None
        # For robust comparison, convert column to string for string operations,
        # and attempt numeric conversion for numeric operations.

        col_as_str = df[column_name].astype(str)
        search_value_lower = search_value.lower()

        if match_type == 'contains':
            filtered_df = df[col_as_str.str.lower().str.contains(search_value_lower, case=False, na=False)]
        elif match_type == 'exact': # Changed from 'equals'
            try: # Attempt numeric exact match
                num_search_value = pd.to_numeric(search_value)
                # Coerce errors in column to NaN, then compare. NaN will not equal num_search_value.
                filtered_df = df[pd.to_numeric(df[column_name], errors='coerce') == num_search_value]
            except ValueError: # Fallback to string exact match
                filtered_df = df[col_as_str.str.lower() == search_value_lower]
        elif match_type == 'not_contains':
            filtered_df = df[~col_as_str.str.lower().str.contains(search_value_lower, case=False, na=False)]
        elif match_type == 'not_exact': # Changed from 'not_equals'
            try: # Attempt numeric non-equality
                num_search_value = pd.to_numeric(search_value)
                # For non-equality, NaNs in the column should also be included unless explicitly handled.
                # Here, they won't match num_search_value, so they are effectively "not equal".
                filtered_df = df[pd.to_numeric(df[column_name], errors='coerce') != num_search_value]
            except ValueError: # Fallback to string non-equality
                filtered_df = df[col_as_str.str.lower() != search_value_lower]
        elif match_type == 'greater_than':
            try:
                num_search_value = pd.to_numeric(search_value)
                filtered_df = df[pd.to_numeric(df[column_name], errors='coerce') > num_search_value]
            except ValueError:
                return "قيمة البحث لـ 'أكبر من' يجب أن تكون رقمية."
        elif match_type == 'less_than':
            try:
                num_search_value = pd.to_numeric(search_value)
                filtered_df = df[pd.to_numeric(df[column_name], errors='coerce') < num_search_value]
            except ValueError:
                return "قيمة البحث لـ 'أصغر من' يجب أن تكون رقمية."
        elif match_type == 'starts_with':
            filtered_df = df[col_as_str.str.startswith(search_value, case=False, na=False)]
        elif match_type == 'ends_with':
            filtered_df = df[col_as_str.str.endswith(search_value, case=False, na=False)]
        else:
            return f"نوع المطابقة '{match_type}' غير معروف."

        if filtered_df is None: # Should ideally not happen if all paths lead to assignment or error return
             logger.error(f"filtered_df remained None for match_type '{match_type}' - this indicates a logic flaw.")
             return "خطأ في تطبيق الفلتر. لم يتم تحديد نتائج."

        logger.info(f"Search for '{search_value}' in column '{column_name}' with match type '{match_type}' yielded {len(filtered_df)} results.")
        return filtered_df

    except HttpError as error:
        logger.error(f"HttpError during file download for search (File ID: {file_id}): {error}")
        return "خطأ أثناء تنزيل الملف للبحث."
    except Exception as e:
        logger.error(f"General error in perform_actual_search_sync (File ID: {file_id}): {e}", exc_info=True)
        return f"خطأ عام وغير متوقع أثناء البحث: {type(e).__name__}"

async def perform_actual_search_async(gdrive_service, file_id: str, mime_type: str, column_name: str, match_type: str, search_value: str) -> pd.DataFrame | str:
    return await asyncio.to_thread(perform_actual_search_sync, gdrive_service, file_id, mime_type, column_name, match_type, search_value)

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

# This is the old get_openai_response, which will be replaced by the one defined above.
# async def get_openai_response(api_key: str, user_question: str) -> str:
#     try:
#         logger.info(f"Sending request to OpenAI API with question: {user_question}")
#         def generate_sync():
#             client = OpenAI(api_key=api_key)
#             completion = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant. Please respond in Arabic unless the user asks for another language."},
#                     {"role": "user", "content": user_question}
#                 ],
#                 max_tokens=300
#             )
#             return completion.choices[0].message.content.strip()
#         assistant_reply = await asyncio.to_thread(generate_sync)
#         logger.info("Received response from OpenAI API.")
#         return assistant_reply if assistant_reply else "لم أتلق ردًا نصيًا من OpenAI."
#     except Exception as e:
#         logger.error(f"Error communicating with OpenAI API: {e}")
#         error_message = str(e).lower()
#         if "invalid api key" in error_message or "incorrect api key" in error_message:
#              return "عذراً، مفتاح OpenAI API غير صالح. يرجى التحقق منه."
#         if "quota" in error_message or "rate limit" in error_message:
#             return "عذراً، لقد تجاوزت حصة الاستخدام أو الحد الأقصى للطلبات لـ OpenAI API. يرجى المحاولة لاحقًا."
#         return f"عذراً، حدث خطأ أثناء محاولة الاتصال بـ OpenAI: ({type(e).__name__})"

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
        # Construct messages list for the new get_openai_response
        messages_for_testai = [
            {"role": "system", "content": "أنت مساعد مفيد. الرجاء الرد باللغة العربية ما لم يطلب المستخدم لغة أخرى."},
            {"role": "user", "content": user_question}
        ]
        reply_text = await get_openai_response(OPENAI_API_KEY, messages_for_testai)
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
        if update.message and update.message.text and update.message.text.startswith("❓ اسأل عن بياناتي"):
            await update.message.reply_text("ما هو سؤالك عن البيانات؟ يمكنك استخدام الصيغة: `/askdata سؤالك هنا --file اسم_الملف.csv` لتحديد ملف معين.")
            return
        await update.message.reply_text("الرجاء إدخال سؤالك بعد الأمر. مثال: `/askdata ما هو متوسط المبيعات --file data.csv`")
        return

    full_text_args = " ".join(context.args)

    # Parse question and filename
    user_question_text = full_text_args
    target_file_name_query = None
    match = re.search(r"(.+?)\s*--file\s+(.+)", full_text_args, re.IGNORECASE)
    if match:
        user_question_text = match.group(1).strip()
        target_file_name_query = match.group(2).strip()

    if not user_question_text: # Only --file was provided, or empty question
        await update.message.reply_text("الرجاء إدخال سؤالك قبل `--file`. مثال: `/askdata ما هو متوسط المبيعات --file data.csv`")
        return

    logger.info(f"User {update.effective_user.id} asked via /askdata: Question='{user_question_text}', File Query='{target_file_name_query}'")
    thinking_message = await update.message.reply_text(f"لحظات، أفكر في سؤالك: \"{escape_markdown_v2(user_question_text)}\"...")

    if not OPENAI_API_KEY:
        await context.bot.edit_message_text(chat_id=thinking_message.chat_id, message_id=thinking_message.message_id, text="مفتاح OpenAI API غير مُعد.")
        logger.error("OPENAI_API_KEY is not set.")
        return

    target_file_id_for_openai = None
    target_mime_type_for_openai = None
    schema_description_for_openai = None
    file_context_message = "" # Additional context for OpenAI based on file

    if target_file_name_query:
        await context.bot.edit_message_text(chat_id=thinking_message.chat_id, message_id=thinking_message.message_id, text=f"جاري البحث عن ملفك '{escape_markdown_v2(target_file_name_query)}' في Google Drive...")
        service = await get_gdrive_service_async()
        if not service:
            await context.bot.edit_message_text(chat_id=thinking_message.chat_id, message_id=thinking_message.message_id, text="تعذر الاتصال بخدمة Google Drive للبحث عن الملف.")
            # Fallback to answering without file context
        else:
            found_file_id, found_file_name, found_mime_type = await find_file_in_gdrive_async(service, target_file_name_query)

            if found_file_id:
                await context.bot.edit_message_text(chat_id=thinking_message.chat_id, message_id=thinking_message.message_id, text=f"تم العثور على ملف '{escape_markdown_v2(found_file_name)}'. جاري استخلاص مخطط البيانات...")

                # Check MIME type for compatibility (CSV/Excel)
                compatible_mime_types = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
                if found_mime_type in compatible_mime_types:
                    target_file_id_for_openai = found_file_id
                    target_mime_type_for_openai = found_mime_type

                    schema_description_for_openai = await get_schema_from_file_async(service, target_file_id_for_openai, target_mime_type_for_openai)

                    if schema_description_for_openai:
                        file_context_message = f"\n\nتم العثور على ملف باسم '{escape_markdown_v2(found_file_name)}' من نوع '{target_mime_type_for_openai}'. {schema_description_for_openai}."
                        await context.bot.edit_message_text(chat_id=thinking_message.chat_id, message_id=thinking_message.message_id, text=f"تم استخلاص مخطط البيانات لملف '{escape_markdown_v2(found_file_name)}'. الآن أفكر في إجابة سؤالك...")
                    else:
                        file_context_message = f"\n\nتم العثور على ملف باسم '{escape_markdown_v2(found_file_name)}' ولكن لم أتمكن من استخلاص مخطط الأعمدة. سأحاول الإجابة على سؤالك بدونه."
                        await context.bot.edit_message_text(chat_id=thinking_message.chat_id, message_id=thinking_message.message_id, text=file_context_message)
                else:
                    file_context_message = f"\n\nالملف '{escape_markdown_v2(found_file_name)}' (نوع: {found_mime_type}) ليس من نوع CSV أو Excel. سأجيب على سؤالك بدون تحليل محتواه."
                    await context.bot.edit_message_text(chat_id=thinking_message.chat_id, message_id=thinking_message.message_id, text=file_context_message)
            else:
                file_context_message = f"\n\nلم أتمكن من العثور على ملف بالاسم '{escape_markdown_v2(target_file_name_query)}' في Google Drive. سأحاول الإجابة على سؤالك بدون سياق الملف."
                await context.bot.edit_message_text(chat_id=thinking_message.chat_id, message_id=thinking_message.message_id, text=file_context_message)

    # Construct the messages list for OpenAI
    system_message_content = "أنت مساعد ذكاء اصطناعي متخصص في تحليل البيانات والإجابة على الأسئلة المتعلقة بها. "
    if schema_description_for_openai: # This now implies a file was found and schema extracted
        system_message_content += (
            f"المستخدم يسأل عن بيانات في ملف. {schema_description_for_openai}. "
            "الرجاء استخدام هذه الأعمدة كمرجع أساسي لفهم البيانات وتقديم إجابة دقيقة أو اقتراح كود Pandas مناسب إذا كان ذلك ملائمًا للسؤال. "
            "إذا كان السؤال يتطلب البحث في البيانات نفسها (مثلاً، 'ما هو متوسط السعر؟')، وضح أنك لا تستطيع الوصول المباشر لمحتوى الملف ولكن يمكنك المساعدة في كيفية إجراء الحساب بناءً على الأعمدة المعطاة."
        )
    elif target_file_name_query: # File was specified but not found, or schema not extracted
         system_message_content += (
             f"حاول المستخدم تحديد ملف باسم '{escape_markdown_v2(target_file_name_query)}' ولكن لم يتم العثور عليه أو لم يتم استخلاص مخططه. "
             "أجب على السؤال بشكل عام قدر الإمكان، أو اطلب من المستخدم التأكد من اسم الملف أو تحميله مباشرة."
         )
    else: # No file specified
        system_message_content += "المستخدم يسأل سؤالاً عاماً عن البيانات. "

    system_message_content += (
        "حاول الإجابة على سؤاله بشكل مباشر ومفيد. "
        "إذا كان السؤال يتطلب عملية حسابية معقدة أو البحث في ملف بيانات ولم يكن لديك وصول مباشر للملف (أو لم يتم تحديد ملف صالح)، "
        "يمكنك توضيح الخطوات أو نوع الاستعلام الذي قد يحتاجه المستخدم. لا تخترع بيانات إذا لم تكن متوفرة."
    )

    messages_for_askdata = [
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": user_question_text}
    ]

    logger.info(f"Sending to OpenAI for /askdata with messages: {messages_for_askdata}")
    response = await get_openai_response(OPENAI_API_KEY, messages_for_askdata)

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
async def list_gdrive_files_command(update: Update, context: ContextTypes.DEFAULT_TYPE, page_token: str = None, folder_id: str = 'root'):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_callback = hasattr(update, 'callback_query') and update.callback_query is not None
    message_to_edit_id = None

    if is_callback:
        message_to_edit_id = update.callback_query.message.message_id
        # Ensure query is answered for callbacks
        try:
            await update.callback_query.answer()
        except Exception as e_ans: # Can fail if already answered or too old
            logger.debug(f"Callback query answer failed (likely already answered): {e_ans}")
    else: # Initial command call
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    if not os.path.exists(CLIENT_SECRET_FILE):
        error_msg = f"ملف `{CLIENT_SECRET_FILE}` غير موجود."
        if is_callback and message_to_edit_id: await context.bot.edit_message_text(chat_id=chat_id, message_id=message_to_edit_id, text=error_msg, parse_mode='MarkdownV2')
        else: await context.bot.send_message(chat_id=chat_id, text=error_msg, parse_mode='MarkdownV2')
        logger.error(f"Missing {CLIENT_SECRET_FILE} at {os.getcwd()}")
        return

    if not os.path.exists(TOKEN_FILE) and not is_callback : # Only prompt for auth on direct command, not on page navigation
        await context.bot.send_message(chat_id, "للوصول لـ Google Drive، أحتاج إذنك (مرة واحدة). اتبع التعليمات في الطرفية.")
        # Actual auth flow happens in get_gdrive_service_async if token is missing/invalid

    service = await get_gdrive_service_async()
    if not service:
        error_msg = "لم أتمكن من الاتصال بـ Google Drive."
        if is_callback and message_to_edit_id: await context.bot.edit_message_text(chat_id=chat_id, message_id=message_to_edit_id, text=error_msg)
        else: await context.bot.send_message(chat_id=chat_id, text=error_msg)
        return

    try:
        logger.info(f"User {user_id} requested GDrive files. Folder: '{folder_id}', PageToken: '{page_token}'")

        drive_query = f"'{folder_id}' in parents and trashed=false"
        page_size = 10 # Number of files per page

        def list_files_sync():
            return service.files().list(
                q=drive_query,
                pageSize=page_size,
                fields="nextPageToken, files(id, name, mimeType, webViewLink)",
                orderBy="folder, name", # Folders first, then by name
                pageToken=page_token
            ).execute()

        results = await asyncio.to_thread(list_files_sync)
        items = results.get('files', [])
        next_page_token_from_api = results.get('nextPageToken')

        message_text_parts = []
        keyboard_buttons_rows = [] # Each element is a row of buttons

        if not items and not page_token: # Only show "empty" if it's the first page and no items
            message_text_parts.append("المجلد فارغ أو لا يحتوي على ملفات يمكن الوصول إليها.")
        else:
            if not is_callback: # For initial call, send a header message
                 message_text_parts.append("أحدث الملفات والمجلدات:") # This might be replaced if editing

            for item in items:
                icon = "📁" if item['mimeType'] == 'application/vnd.google-apps.folder' else "📄"
                link = item.get('webViewLink', '') # Not used directly in message to save space, but good for logs
                file_name_escaped = escape_markdown_v2(item['name'])

                # Each file/folder will have its own message or be part of a list in one message.
                # For simplicity with inline buttons, sending one message per item is easier if buttons are complex.
                # However, to use pagination buttons effectively, all items for a page should be in ONE message.

                file_info_line = f"{icon} {file_name_escaped}"
                message_text_parts.append(file_info_line)

                action_buttons = []
                if item['mimeType'] != 'application/vnd.google-apps.folder':
                    action_buttons.append(InlineKeyboardButton("📥 تحميل", callback_data=f'download_gdrive_{item["id"]}'))

                if item['mimeType'] == 'text/csv':
                    action_buttons.append(InlineKeyboardButton("📄 اقرأ CSV", callback_data=f'read_csv_{item["id"]}'))

                if item['mimeType'] == 'application/json' or item['name'].lower().endswith('.json'):
                    action_buttons.append(InlineKeyboardButton("📊 تحليل JSON", callback_data=f'analyze_json_gdrive_{item["id"]}'))

                if item['mimeType'] == 'text/plain' or item['name'].lower().endswith('.txt'):
                    action_buttons.append(InlineKeyboardButton("📖 عرض TXT", callback_data=f'options_txt_gdrive_{item["id"]}'))

                if action_buttons:
                    keyboard_buttons_rows.append(action_buttons) # Add row of buttons for this file

        # Pagination buttons
        pagination_row = []
        # Simplified: No "Previous" button for now due to GDrive API limitations (no prevPageToken)
        # If we were on page > 1 (i.e., page_token was not None), a "Previous" button could be constructed
        # if we stored the token that led to the current page.

        if next_page_token_from_api:
            pagination_row.append(InlineKeyboardButton("التالي ⏪", callback_data=f"gdrive_page_{folder_id}_{next_page_token_from_api}"))

        if pagination_row:
            keyboard_buttons_rows.append(pagination_row)

        final_message_text = "\n".join(message_text_parts)
        if not final_message_text: # Should not happen if logic is correct
            final_message_text = "لا توجد عناصر لعرضها."

        final_reply_markup = InlineKeyboardMarkup(keyboard_buttons_rows) if keyboard_buttons_rows else None

        if is_callback and message_to_edit_id:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_to_edit_id,
                text=final_message_text,
                reply_markup=final_reply_markup,
                parse_mode='MarkdownV2'
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=final_message_text,
                reply_markup=final_reply_markup,
                parse_mode='MarkdownV2'
            )

    except HttpError as http_err:
        logger.error(f'HttpError listing GDrive files: {http_err}. Response: {http_err.content}')
        error_message = f'خطأ في عرض ملفات GDrive \\({http_err.resp.status}\\)\\. قد تحتاج إلى إعادة المصادقة إذا انتهت صلاحية الرمز المميز\\. جرب حذف `token\\.json` وإعادة تشغيل الأمر\\.'
        if is_callback and message_to_edit_id: await context.bot.edit_message_text(chat_id=chat_id, message_id=message_to_edit_id, text=error_message, parse_mode='MarkdownV2')
        else: await context.bot.send_message(chat_id=chat_id, text=error_message, parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f'Error listing GDrive files: {e}', exc_info=True)
        error_message = f'خطأ غير متوقع في عرض ملفات GDrive: {escape_markdown_v2(str(type(e).__name__))}'
        if is_callback and message_to_edit_id: await context.bot.edit_message_text(chat_id=chat_id, message_id=message_to_edit_id, text=error_message, parse_mode='MarkdownV2')
        else: await context.bot.send_message(chat_id=chat_id, text=error_message, parse_mode='MarkdownV2')


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
        # The escape_markdown_v2 function is already globally defined, so we can use it directly.
        # def escape_markdown_v2(text: str) -> str:
        #     escape_chars = r'_*[]()~`>#+-.!{}='
        #     return "".join(f'\\{char}' if char in escape_chars else char for char in text)

        await context.bot.send_message(chat_id=chat_id_to_reply, text=escape_markdown_v2(analysis_result), parse_mode='MarkdownV2')

    elif callback_data.startswith('download_gdrive_'):
        file_id = callback_data.split('_', 2)[2]
        logger.info(f"User {user_id} requested to download GDrive file ID: {file_id}")
        original_message_text = query.message.text if query.message else "تجهيز الملف..."

        try:
            await query.edit_message_text(text=f"{original_message_text}\n\n---\n📥 جاري تجهيز الملف للتنزيل...")
        except Exception as e:
            logger.warning(f"Could not edit original button message for download_gdrive_ action: {e}")
            await context.bot.send_message(chat_id=chat_id_to_reply, text="📥 جاري تجهيز الملف للتنزيل...")

        await context.bot.send_chat_action(chat_id=chat_id_to_reply, action=ChatAction.UPLOAD_DOCUMENT)

        service = await get_gdrive_service_async()
        if not service:
            await context.bot.send_message(chat_id=chat_id_to_reply, text="فشل الاتصال بـ Google Drive. لا يمكن تنزيل الملف.")
            # Attempt to revert the "جاري التجهيز" message if possible, or send a new one
            try:
                await query.edit_message_text(text=original_message_text + "\n\n---\n❌ فشل الاتصال بالخدمة.")
            except: pass # Ignore if editing fails
            return

        def download_gdrive_file_sync(gdrive_service, gdrive_file_id_sync) -> tuple[str | None, bytes | None]:
            try:
                file_metadata = gdrive_service.files().get(fileId=gdrive_file_id_sync, fields='name, mimeType').execute()
                original_file_name_sync = file_metadata.get('name')
                # mime_type_sync = file_metadata.get('mimeType') # mime_type not strictly needed for sending

                request = gdrive_service.files().get_media(fileId=gdrive_file_id_sync)
                file_content_stream = io.BytesIO()
                downloader = MediaIoBaseDownloader(file_content_stream, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.info(f"GDrive Download {gdrive_file_id_sync}: {int(status.progress() * 100)}%.")

                file_bytes_sync = file_content_stream.getvalue()
                logger.info(f"Successfully downloaded GDrive file ID: {gdrive_file_id_sync}, Name: {original_file_name_sync}")
                return original_file_name_sync, file_bytes_sync
            except HttpError as http_err_sync:
                logger.error(f"HttpError downloading GDrive file {gdrive_file_id_sync}: {http_err_sync}")
                return None, None
            except Exception as e_sync:
                logger.error(f"General error downloading GDrive file {gdrive_file_id_sync}: {e_sync}")
                return None, None

        original_file_name, file_bytes = await asyncio.to_thread(download_gdrive_file_sync, service, file_id)

        if original_file_name and file_bytes:
            try:
                await context.bot.send_document(chat_id=chat_id_to_reply, document=file_bytes, filename=original_file_name)
                # Try to edit the original message to indicate completion or remove buttons
                await query.edit_message_text(text=f"{original_message_text}\n\n---\n✅ تم إرسال الملف: {escape_markdown_v2(original_file_name)}")
            except Exception as send_exc:
                logger.error(f"Error sending document or editing message after download for GDrive file ID {file_id}: {send_exc}")
                await context.bot.send_message(chat_id=chat_id_to_reply, text=f"تم تنزيل الملف '{escape_markdown_v2(original_file_name)}' ولكن حدث خطأ أثناء إرساله لك. حاول مرة أخرى أو تحقق من الرسائل الخاصة.")
        else:
            await context.bot.send_message(chat_id=chat_id_to_reply, text="عذرًا، لم أتمكن من تنزيل الملف المحدد من Google Drive.")
            # Attempt to revert the "جاري التجهيز" message
            try:
                await query.edit_message_text(text=original_message_text + "\n\n---\n❌ فشل تنزيل الملف.")
            except: pass

    elif callback_data.startswith('analyze_json_gdrive_'):
        file_id = callback_data.split('_', 3)[3] # analyze_json_gdrive_FILEID
        logger.info(f"User {user_id} requested to analyze GDrive JSON file ID: {file_id}")
        original_message_text = query.message.text if query.message else "تحليل JSON..."

        try:
            await query.edit_message_text(text=f"{original_message_text}\n\n---\n📊 جاري تحليل ملف JSON...")
        except Exception as e:
            logger.warning(f"Could not edit original button message for analyze_json_gdrive_ action: {e}")
            await context.bot.send_message(chat_id=chat_id_to_reply, text="📊 جاري تحليل ملف JSON...")

        await context.bot.send_chat_action(chat_id=chat_id_to_reply, action=ChatAction.TYPING)

        service = await get_gdrive_service_async()
        if not service:
            await context.bot.send_message(chat_id=chat_id_to_reply, text="فشل الاتصال بـ Google Drive. لا يمكن تحليل الملف.")
            try: await query.edit_message_text(text=original_message_text + "\n\n---\n❌ فشل الاتصال بالخدمة.")
            except: pass
            return

        def process_json_file_sync(gdrive_service, gdrive_file_id_sync: str) -> str:
            try:
                file_metadata = gdrive_service.files().get(fileId=gdrive_file_id_sync, fields='name, size').execute()
                original_file_name_sync = file_metadata.get('name', 'unknown.json')
                file_size = int(file_metadata.get('size', 0))

                # Limit file size to 5MB for now
                if file_size > 5 * 1024 * 1024: # 5MB
                    logger.warning(f"JSON file {gdrive_file_id_sync} ('{original_file_name_sync}') is too large: {file_size} bytes.")
                    return f"ملف JSON '{escape_markdown_v2(original_file_name_sync)}' كبير جدًا للتحليل المباشر ({file_size / (1024*1024):.2f} MB)."

                request = gdrive_service.files().get_media(fileId=gdrive_file_id_sync)
                file_content_stream = io.BytesIO()
                downloader = MediaIoBaseDownloader(file_content_stream, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    # logger.info(f"JSON Download {gdrive_file_id_sync}: {int(status.progress() * 100)}%.") # Can be verbose

                file_bytes_sync = file_content_stream.getvalue()

                try:
                    json_string = file_bytes_sync.decode('utf-8')
                except UnicodeDecodeError:
                    logger.error(f"UnicodeDecodeError for JSON file {gdrive_file_id_sync} ('{original_file_name_sync}').")
                    return f"خطأ في فك ترميز الملف '{escape_markdown_v2(original_file_name_sync)}'. تأكد أنه بصيغة UTF-8."

                try:
                    data = json.loads(json_string)
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSONDecodeError for file {gdrive_file_id_sync} ('{original_file_name_sync}'): {json_err}")
                    return f"خطأ في تحليل JSON للملف '{escape_markdown_v2(original_file_name_sync)}':\n`{escape_markdown_v2(str(json_err))}`"

                analysis_summary = f"*ملف JSON: {escape_markdown_v2(original_file_name_sync)}*\n"

                if isinstance(data, list):
                    try:
                        df = pd.json_normalize(data)
                        # df = pd.DataFrame(data) # Alternative for simple list of dicts
                        analysis_summary += f"تم تحويل القائمة إلى جدول بيانات \\({len(df)} صفوف, {len(df.columns)} أعمدة\\)\\.\n"
                        analysis_summary += f"*الأعمدة:* `{escape_markdown_v2(', '.join(df.columns))}`\n"
                        if not df.empty:
                             analysis_summary += f"*أول 3 صفوف:*\n```\n{df.head(3).to_string()}\n```"
                        else:
                            analysis_summary += "الجدول الناتج فارغ."
                    except Exception as e_df:
                        logger.error(f"Error converting JSON list to DataFrame for file '{original_file_name_sync}': {e_df}")
                        analysis_summary += "القائمة معقدة ولم يتم تحويلها مباشرة إلى جدول قياسي\\. قد تحتاج إلى تحديد مسار معين للبيانات داخل القائمة\\."

                elif isinstance(data, dict):
                    analysis_summary += "الملف يحتوي على كائن JSON\\. *المفاتيح الرئيسية:* `" + escape_markdown_v2(", ".join(data.keys())) + "`\n"
                    # Try to find a list of records within the dictionary
                    processed_nested_list = False
                    for key, value in data.items():
                        if isinstance(value, list) and value and all(isinstance(i, dict) for i in value):
                            try:
                                df = pd.json_normalize(value)
                                analysis_summary += f"\nتم العثور على قائمة داخل المفتاح '{escape_markdown_v2(key)}' وتحويلها إلى جدول \\({len(df)} صفوف, {len(df.columns)} أعمدة\\):\n"
                                analysis_summary += f"*الأعمدة:* `{escape_markdown_v2(', '.join(df.columns))}`\n"
                                if not df.empty:
                                    analysis_summary += f"*أول 3 صفوف:*\n```\n{df.head(3).to_string()}\n```\n"
                                else:
                                    analysis_summary += "الجدول الناتج من القائمة المتداخلة فارغ.\n"
                                processed_nested_list = True
                                break # Process first such list for now
                            except Exception as e_df_nested:
                                logger.error(f"Error converting nested list under key '{key}' to DataFrame for file '{original_file_name_sync}': {e_df_nested}")
                                analysis_summary += f"تعذر تحويل القائمة المتداخلة تحت المفتاح '{escape_markdown_v2(key)}' إلى جدول قياسي.\n"
                    if not processed_nested_list:
                         analysis_summary += "لم يتم العثور على قائمة بسيطة من السجلات داخل الكائن لتحويلها لجدول تلقائيًا."
                else:
                    analysis_summary += "صيغة JSON غير مدعومة للتحليل المتقدم حاليًا \\(ليست قائمة أو كائن قياسي\\)\\."

                return analysis_summary

            except HttpError as http_err_sync:
                logger.error(f"HttpError processing JSON file {gdrive_file_id_sync}: {http_err_sync}")
                return f"خطأ في الوصول لملف JSON (ID: {gdrive_file_id_sync}) على Google Drive."
            except Exception as e_sync:
                logger.error(f"General error processing JSON file {gdrive_file_id_sync}: {e_sync}")
                return f"خطأ عام أثناء معالجة ملف JSON (ID: {gdrive_file_id_sync}): {type(e_sync).__name__}"

        analysis_result = await asyncio.to_thread(process_json_file_sync, service, file_id)

        try:
            await query.edit_message_text(text=analysis_result, parse_mode='MarkdownV2')
        except Exception as e_edit: # If message is too long or contains problematic markdown
            logger.warning(f"Failed to edit message with JSON analysis, sending as new: {e_edit}")
            await context.bot.send_message(chat_id=chat_id_to_reply, text=analysis_result, parse_mode='MarkdownV2')

    elif callback_data.startswith('options_txt_gdrive_'):
        file_id = callback_data.split('_', 3)[3] # options_txt_gdrive_FILEID
        logger.info(f"User {user_id} requested options for GDrive TXT file ID: {file_id}")

        # Fetch file name to display in the options message
        service = await get_gdrive_service_async()
        if not service:
            await query.edit_message_text(text="فشل الاتصال بـ Google Drive. لا يمكن عرض خيارات الملف.")
            return

        try:
            file_metadata = await asyncio.to_thread(service.files().get(fileId=file_id, fields='name').execute)
            original_file_name = file_metadata.get('name', 'file.txt')
        except Exception as e_meta:
            logger.error(f"Error fetching metadata for TXT options (ID: {file_id}): {e_meta}")
            await query.edit_message_text(text="خطأ في جلب معلومات الملف لعرض الخيارات.")
            return

        txt_options_keyboard = [
            [InlineKeyboardButton(" عرض أول 100 سطر", callback_data=f"view_txt_head_{file_id}")],
            [InlineKeyboardButton(" عرض آخر 100 سطر", callback_data=f"view_txt_tail_{file_id}")],
            # [InlineKeyboardButton("بحث عن كلمة (قريبًا)", callback_data=f"search_txt_{file_id}_disabled")]
        ]
        options_markup = InlineKeyboardMarkup(txt_options_keyboard)
        try:
            await query.edit_message_text(
                text=f"اختر إجراء لملف TXT: '{escape_markdown_v2(original_file_name)}'",
                reply_markup=options_markup,
                parse_mode='MarkdownV2'
            )
        except Exception as e_edit_options:
            logger.error(f"Error editing message to show TXT options: {e_edit_options}")
            # Fallback if edit fails (e.g. message too old, or content identical)
            await context.bot.send_message(
                chat_id=chat_id_to_reply,
                text=f"اختر إجراء لملف TXT: '{escape_markdown_v2(original_file_name)}'",
                reply_markup=options_markup,
                parse_mode='MarkdownV2'
            )

    elif callback_data.startswith('view_txt_head_') or callback_data.startswith('view_txt_tail_'):
        mode = 'head' if callback_data.startswith('view_txt_head_') else 'tail'
        file_id = callback_data.split('_', 3)[3] # view_txt_head_FILEID or view_txt_tail_FILEID

        logger.info(f"User {user_id} requested to view {mode} of GDrive TXT file ID: {file_id}")

        # It's better to send a new message for the processing status,
        # rather than editing the message that contains the options keyboard.
        status_message = await context.bot.send_message(chat_id=chat_id_to_reply, text=f"📖 جاري معالجة ملف TXT ({mode})...")
        await context.bot.send_chat_action(chat_id=chat_id_to_reply, action=ChatAction.TYPING)

        service = await get_gdrive_service_async()
        if not service:
            await context.bot.edit_message_text(chat_id=status_message.chat_id, message_id=status_message.message_id, text="فشل الاتصال بـ Google Drive.")
            return

        def process_txt_file_sync(gdrive_service, gdrive_file_id_sync: str, view_mode: str) -> str:
            try:
                file_metadata = gdrive_service.files().get(fileId=gdrive_file_id_sync, fields='name, size').execute()
                original_file_name_sync = file_metadata.get('name', 'file.txt')
                file_size = int(file_metadata.get('size', 0))

                # Limit file size for viewing (e.g., 2MB)
                if file_size > 2 * 1024 * 1024: # 2MB
                    logger.warning(f"TXT file {gdrive_file_id_sync} ('{original_file_name_sync}') is too large for viewing: {file_size} bytes.")
                    return f"ملف TXT '{escape_markdown_v2(original_file_name_sync)}' كبير جدًا للعرض المباشر ({file_size / (1024*1024):.2f} MB)."

                request = gdrive_service.files().get_media(fileId=gdrive_file_id_sync)
                file_content_stream = io.BytesIO()
                downloader = MediaIoBaseDownloader(file_content_stream, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()

                file_bytes_sync = file_content_stream.getvalue()

                try:
                    # Try common encodings, starting with utf-8
                    encodings_to_try = ['utf-8', 'iso-8859-1', 'windows-1256'] # windows-1256 for Arabic
                    text_content = None
                    for enc in encodings_to_try:
                        try:
                            text_content = file_bytes_sync.decode(enc)
                            logger.info(f"Decoded TXT file {gdrive_file_id_sync} with {enc}")
                            break
                        except UnicodeDecodeError:
                            logger.debug(f"Failed to decode TXT {gdrive_file_id_sync} with {enc}")

                    if text_content is None:
                        logger.error(f"Could not decode TXT file {gdrive_file_id_sync} with common encodings.")
                        return f"خطأ في فك ترميز الملف '{escape_markdown_v2(original_file_name_sync)}'. قد يكون الترميز غير مدعوم."

                except UnicodeDecodeError: # Should be caught by the loop above
                    logger.error(f"UnicodeDecodeError for TXT file {gdrive_file_id_sync} ('{original_file_name_sync}').")
                    return f"خطأ في فك ترميز الملف '{escape_markdown_v2(original_file_name_sync)}'."

                lines = text_content.splitlines()

                if view_mode == 'head':
                    selected_lines = lines[:100]
                    result_header = f"أول 100 سطر من ملف '{escape_markdown_v2(original_file_name_sync)}':\n"
                else: # tail
                    selected_lines = lines[-100:]
                    result_header = f"آخر 100 سطر من ملف '{escape_markdown_v2(original_file_name_sync)}':\n"

                output_text = "\n".join(selected_lines)

                if not output_text.strip():
                    return f"{result_header}\n(لا يوجد محتوى لعرضه في هذا الجزء من الملف)"

                # Truncate if too long for a Telegram message (4096 chars limit)
                max_len = 4000 # Leave some room for header and markdown code block
                if len(output_text) > max_len:
                    output_text = output_text[:max_len] + "\n... (تم اقتطاع النص لطوله الزائد)"

                return f"{result_header}```\n{output_text}\n```" # Using Markdown code block

            except HttpError as http_err_sync:
                logger.error(f"HttpError processing TXT file {gdrive_file_id_sync}: {http_err_sync}")
                return f"خطأ في الوصول لملف TXT (ID: {gdrive_file_id_sync}) على Google Drive."
            except Exception as e_sync:
                logger.error(f"General error processing TXT file {gdrive_file_id_sync}: {e_sync}")
                return f"خطأ عام أثناء معالجة ملف TXT (ID: {gdrive_file_id_sync}): {type(e_sync).__name__}"

        processed_text = await asyncio.to_thread(process_txt_file_sync, service, file_id, mode)

        # Edit the "جاري المعالجة" message with the result or send a new one if it's too long / causes issues
        try:
             await context.bot.edit_message_text(chat_id=status_message.chat_id, message_id=status_message.message_id, text=processed_text, parse_mode='MarkdownV2')
        except Exception as e_final_edit:
            logger.warning(f"Failed to edit status message with TXT content, sending as new: {e_final_edit}")
            await context.bot.send_message(chat_id=chat_id_to_reply, text=processed_text, parse_mode='MarkdownV2')

    elif callback_data.startswith('gdrive_page_'):
        # Format: gdrive_page_FOLDERID_PAGETOKEN
        # If PAGETOKEN is 'None' (as string), it means first page of that FOLDERID
        parts = callback_data.split('_', 3)
        folder_id_from_cb = parts[2]
        next_token_from_cb = parts[3] if len(parts) > 3 else None
        if next_token_from_cb == 'None': # Handle explicit 'None' string if passed
            next_token_from_cb = None

        logger.info(f"User {user_id} requested GDrive page. Folder ID: {folder_id_from_cb}, Next Page Token: {next_token_from_cb}")
        # Call list_gdrive_files_command, making sure `update` is the CallbackQuery object
        await list_gdrive_files_command(update.callback_query, context, page_token=next_token_from_cb, folder_id=folder_id_from_cb)


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

    # --- Add Search Conversation Handler ---
    search_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('searchdata', start_search_conversation)],
        states={
            SELECT_FILE: [MessageHandler(filters.TEXT & ~filters.COMMAND, received_filename_for_search)],
            SELECT_COLUMN: [CallbackQueryHandler(received_column_for_search, pattern=r'^search_col_select_\d+$')],
            SELECT_MATCH_TYPE: [CallbackQueryHandler(received_match_type, pattern=r'^searchmatch_')], # Pattern example
            INPUT_SEARCH_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, received_search_value)],
        },
        fallbacks=[CommandHandler('cancel_search', cancel_search_conversation)],
        # persistent=False, # Using default (memory-based persistence for now)
        # allow_reentry=True # Default is False, which is usually fine
    )
    application.add_handler(search_conv_handler)

    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo_message))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command)) # Keep this last for unhandled commands
    
    logger.info("Bot is now polling for updates...")
    application.run_polling()

    logger.info("Bot has stopped.")


# --- Search Results Handling ---
async def handle_search_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    search_params = context.user_data

    file_id = search_params.get('search_file_id')
    mime_type = search_params.get('search_mime_type')
    column_name = search_params.get('search_selected_column_name')
    match_type = search_params.get('search_match_type')
    search_value = search_params.get('search_value')
    file_name = search_params.get('search_file_name', 'الملف المحدد')

    if not all([file_id, mime_type, column_name, match_type, search_value is not None]): # search_value can be empty string
        logger.error(f"User {user_id}: Missing search parameters in handle_search_results. Data: {search_params}")
        await update.message.reply_text("حدث خطأ، بعض معلمات البحث مفقودة. يرجى المحاولة مرة أخرى.")
        return

    await update.message.reply_text(
        f"جاري البحث في ملف '{escape_markdown_v2(file_name)}' عن قيمة '{escape_markdown_v2(search_value)}' في عمود '{escape_markdown_v2(column_name)}' بنوع المطابقة '{match_type}'...",
        parse_mode='MarkdownV2'
    )
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    service = await get_gdrive_service_async()
    if not service:
        await update.message.reply_text("تعذر الاتصال بخدمة Google Drive لإجراء البحث.")
        return

    results_df_or_error_str = await perform_actual_search_async(
        service, file_id, mime_type, column_name, match_type, search_value
    )

    if isinstance(results_df_or_error_str, pd.DataFrame):
        results_df = results_df_or_error_str
        logger.info(f"User {user_id}: Search successful. Found {len(results_df)} results in '{file_name}'.")
        # The line below was commented out in the previous correct diff, so keeping it commented.
        # context.user_data['search_results_df'] = results_df

        if results_df.empty:
            await update.message.reply_text("لم يتم العثور على نتائج تطابق معايير البحث المحددة.")
        else:
            num_total_results = len(results_df)
            max_preview_rows = 10
            preview_df = results_df.head(max_preview_rows)

            try:
                preview_text = preview_df.to_string(index=False, na_rep='-')
            except Exception as e_to_str:
                logger.error(f"Error converting DataFrame to string for preview: {e_to_str}")
                preview_text = "خطأ في تنسيق معاينة النتائج."

            max_chars_telegram = 4096
            message_header = f"تم العثور على {num_total_results} نتيجة."
            if num_total_results > 0:
                message_header += f"\nإليك معاينة لأول {min(num_total_results, max_preview_rows)} نتيجة:\n\n"

            code_block_ticks = "\n```\n"

            # Adjusted safety margin for more reliable truncation
            available_chars_for_text = max_chars_telegram - (len(message_header) + len(code_block_ticks) * 2 + 100)

            if len(preview_text) > available_chars_for_text:
                # Ensure we don't cut in the middle of a multi-byte character if string is unicode
                # However, Python string slicing is based on character counts, not bytes.
                # The main concern is the overall message length for Telegram.
                preview_text = preview_text[:available_chars_for_text] + "\n... (تم اقتطاع المعاينة لطولها الزائد)"

            message = f"{message_header}```\n{preview_text}\n```"

            if num_total_results > max_preview_rows:
                message += f"\n\n(يتم عرض أول {max_preview_rows} نتيجة فقط من إجمالي {num_total_results})."
                # TODO: Implement "Download full results as CSV" button here later
                await update.message.reply_text(message, parse_mode='MarkdownV2')
            else:
                await update.message.reply_text(message, parse_mode='MarkdownV2')

    else:
        error_str = results_df_or_error_str
        logger.error(f"User {user_id}: Search failed for file '{file_name}': {error_str}")
        await update.message.reply_text(f"خطأ في البحث: {escape_markdown_v2(error_str)}")


# --- /searchdata Conversation Functions ---
async def start_search_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear() # Clear any previous user_data from this conversation
    await update.message.reply_text(
        "أهلاً بك في معالج البحث المتقدم!\n"
        "الرجاء إدخال اسم ملف CSV أو Excel الذي تريد البحث فيه (أو جزء من اسمه)."
    )
    return SELECT_FILE

async def received_filename_for_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    filename_query = update.message.text.strip()
    if not filename_query:
        await update.message.reply_text("لم يتم إدخال اسم ملف. الرجاء إدخال اسم ملف صالح أو استخدم /cancel_search لإلغاء البحث.")
        return SELECT_FILE

    await update.message.reply_text(f"جاري البحث عن ملف باسم: '{escape_markdown_v2(filename_query)}'...")

    service = await get_gdrive_service_async()
    if not service:
        await update.message.reply_text("تعذر الاتصال بخدمة Google Drive. لا يمكن متابعة البحث. حاول مرة أخرى لاحقًا أو استخدم /cancel_search.")
        return ConversationHandler.END # Or SELECT_FILE to allow retry without full cancel

    found_file_id, found_file_name, found_mime_type = await find_file_in_gdrive_async(service, filename_query)

    compatible_mime_types = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']

    if found_file_id and found_mime_type in compatible_mime_types:
        context.user_data['search_file_id'] = found_file_id
        context.user_data['search_file_name'] = found_file_name
        context.user_data['search_mime_type'] = found_mime_type

        await update.message.reply_text(f"تم العثور على ملف: '{escape_markdown_v2(found_file_name)}' (نوع: {found_mime_type}).\nجاري استخلاص الأعمدة...")

        columns = await get_schema_from_file_async(service, found_file_id, found_mime_type)

        if columns: # Check if columns is not None and not empty
            context.user_data['search_file_columns'] = columns
            keyboard = [[InlineKeyboardButton(col, callback_data=f"search_col_select_{idx}")] for idx, col in enumerate(columns)]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("الرجاء اختيار العمود الذي تريد البحث فيه:", reply_markup=reply_markup)
            return SELECT_COLUMN
        else:
            await update.message.reply_text(
                f"تم العثور على الملف '{escape_markdown_v2(found_file_name)}' ولكن لم أتمكن من قراءة الأعمدة. "
                "الرجاء التأكد من أن الملف غير فارغ، وغير محمي بكلمة مرور، ويحتوي على صف ترويسة صحيح. "
                "حاول مرة أخرى باسم ملف آخر أو استخدم /cancel_search."
            )
            return SELECT_FILE # Stay in the same state to allow user to try another filename
    else:
        if found_file_id: # File found but not compatible type
             await update.message.reply_text(
                f"تم العثور على ملف '{escape_markdown_v2(found_file_name)}' ولكن نوعه ({found_mime_type}) ليس CSV أو Excel. "
                "لا يمكن البحث في هذا النوع من الملفات حاليًا. الرجاء إدخال اسم ملف CSV أو Excel صالح أو استخدم /cancel_search."
            )
        else: # File not found
            await update.message.reply_text(
                "لم يتم العثور على ملف CSV أو Excel بهذا الاسم. "
                "الرجاء التأكد من الاسم والمحاولة مرة أخرى، أو استخدم /cancel_search."
            )
        return SELECT_FILE


async def received_column_for_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Placeholder - Full implementation in next subtask
    query = update.callback_query
    await query.answer()
    selected_col_index = int(query.data.split('_')[-1])
    selected_column_name = context.user_data['search_file_columns'][selected_col_index]
    context.user_data['search_selected_column_name'] = selected_column_name
    context.user_data['search_selected_column_index'] = selected_col_index

    logger.info(f"User selected column '{selected_column_name}' (index {selected_col_index}) for file '{context.user_data.get('search_file_name')}'.")

    await query.edit_message_text(text=f"تم اختيار عمود: '{escape_markdown_v2(selected_column_name)}'.\nالآن، كيف تريد البحث؟ (سيتم عرض خيارات المطابقة)")
    # For now, ending the conversation here. Next step will define SELECT_MATCH_TYPE state.
    # return ConversationHandler.END # Original placeholder end

    match_types = {
        'contains': "يحتوي على (نص)",
        'exact': "يساوي تمامًا (نص/رقم)", # Changed 'equals' to 'exact' for clarity
        'not_contains': "لا يحتوي على (نص)",
        'not_exact': "لا يساوي تمامًا (نص/رقم)", # Changed 'not_equals'
        'greater_than': "أكبر من (رقم)",
        'less_than': "أصغر من (رقم)",
        'starts_with': "يبدأ بـ (نص)",
        'ends_with': "ينتهي بـ (نص)"
    }
    keyboard = []
    # Create two buttons per row for a cleaner look
    row_buttons = []
    for key, text_label in match_types.items():
        row_buttons.append(InlineKeyboardButton(text_label, callback_data=f"search_match_type_{key}"))
        if len(row_buttons) == 2:
            keyboard.append(row_buttons)
            row_buttons = []
    if row_buttons: # Add any remaining button
        keyboard.append(row_buttons)

    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        text=f"تم اختيار عمود: '{escape_markdown_v2(selected_column_name)}'.\nالرجاء اختيار نوع المطابقة للقيمة التي ستبحث عنها:",
        reply_markup=reply_markup,
        parse_mode='MarkdownV2'
    )
    return SELECT_MATCH_TYPE

async def received_match_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Placeholder - Full implementation in a future subtask
    query = update.callback_query
    await query.answer()
    # Storing the selected match type
    match_type_key = query.data.split('_')[-1]
    context.user_data['search_match_type'] = match_type_key

    # For user-friendliness, get the display text of the match type
    # This requires match_types to be accessible here or passed/redefined.
    # For now, just use the key.
    logger.info(f"User selected match type: {match_type_key}")

    await query.edit_message_text(text=f"تم اختيار نوع المطابقة: {match_type_key}.\nالرجاء إدخال القيمة التي تريد البحث عنها في عمود '{escape_markdown_v2(context.user_data.get('search_selected_column_name', 'المحدد'))}'.")
    return INPUT_SEARCH_VALUE # Transition to inputting search value

async def received_search_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    search_value = update.message.text.strip()
    if not search_value:
        await update.message.reply_text("لم يتم إدخال قيمة للبحث. الرجاء إدخال قيمة أو استخدم /cancel_search للإلغاء.")
        return INPUT_SEARCH_VALUE # Stay in the same state to prompt again

    context.user_data['search_value'] = search_value

    logger.info(f"Search parameters collected for user {update.effective_user.id}:")
    logger.info(f"  File ID: {context.user_data.get('search_file_id')}")
    logger.info(f"  File Name: {context.user_data.get('search_file_name')}")
    logger.info(f"  Selected Column Index: {context.user_data.get('search_selected_column_index')}")
    logger.info(f"  Selected Column Name: {context.user_data.get('search_selected_column_name')}")
    logger.info(f"  Match Type: {context.user_data.get('search_match_type')}")
    logger.info(f"  Search Value: {search_value}")

    # Placeholder for actual search logic
    await update.message.reply_text(
        f"تم استلام كل معطيات البحث.\n"
        f"الملف: `{escape_markdown_v2(context.user_data.get('search_file_name', 'غير محدد'))}`\n"
        f"العمود: `{escape_markdown_v2(context.user_data.get('search_selected_column_name', 'غير محدد'))}`\n"
        f"نوع المطابقة: `{escape_markdown_v2(context.user_data.get('search_match_type', 'غير محدد'))}`\n"
        f"القيمة للبحث: `{escape_markdown_v2(search_value)}`\n\n"
        "(ملاحظة: سيتم الآن تنفيذ البحث الفعلي وعرض النتائج \\- هذه الميزة قيد التطوير حاليًا\\.)",
        parse_mode='MarkdownV2'
    )

    # TODO: In the next plan step, call a function here like:
    # await perform_actual_search(update, context) # This is now handle_search_results
    # For now, we just end the conversation after calling handle_search_results.

    await handle_search_results(update, context) # Call the new handler

    context.user_data.clear()
    return ConversationHandler.END

async def cancel_search_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("تم إلغاء عملية البحث. يمكنك البدء من جديد باستخدام /searchdata.")
    context.user_data.clear()
    return ConversationHandler.END