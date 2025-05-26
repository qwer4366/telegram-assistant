# main.py
import logging
import io # لاستخدام stream البايتات في الذاكرة
import qrcode # لإنشاء رموز QR
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# --- إعدادات البوت الأساسية ---
# !!! تحذير: لا ترفع هذا التوكن إلى GitHub إذا كان المستودع عاماً أو مشتركاً !!!
BOT_TOKEN = "7719226402:AAHQ2RMk5e4SFDBhaVxk-hodY9PelRjRVFY"

# المعرفات التي قدمتها (يمكن استخدامها كمعرفات للمسؤولين مثلاً)
ADMIN_ID_1 = 1263152179
ADMIN_ID_2 = 697852646
ADMIN_IDS = [ADMIN_ID_1, ADMIN_ID_2]

# إعداد تسجيل الدخول (مفيد لتتبع الأخطاء)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- دوال معالجة الأوامر والرسائل ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يرسل رسالة ترحيب عند إرسال الأمر /start."""
    user = update.effective_user
    welcome_message = (
        f"أهلاً بك يا {user.first_name}!\n"
        f"أنا بوتك المساعد. أرسل لي أي رسالة نصية وسأرددها.\n"
        f"جرب أيضاً الأمر /admin_test إذا كنت أحد المسؤولين.\n"
        f"لإنشاء QR code، استخدم الأمر: /qr <النص أو الرابط>"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)
    logger.info(f"User {user.id} ({user.first_name}) started the bot.")

async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يردد الرسالة النصية التي يرسلها المستخدم."""
    user_message = update.message.text
    reply_text = f"أنت قلت: {user_message}"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=reply_text)
    logger.info(f"User {update.effective_user.id} sent text: '{user_message}', bot echoed.")

async def admin_test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """أمر تجريبي للمسؤولين فقط."""
    user_id = update.effective_user.id
    if user_id in ADMIN_IDS:
        reply_text = f"أهلاً بالمسؤول {update.effective_user.first_name}! هذا أمر خاص بك."
        logger.info(f"Admin {user_id} executed /admin_test successfully.")
    else:
        reply_text = "عذراً، هذا الأمر مخصص للمسؤولين فقط."
        logger.warning(f"User {user_id} (not an admin) tried to execute /admin_test.")
    await context.bot.send_message(chat_id=update.effective_chat.id, text=reply_text)

# --- دوال ميزة QR Code ---
async def generate_qr_image(text_to_encode: str) -> io.BytesIO | None:
    """
    تنشئ صورة QR Code من النص المدخل وتعيدها كـ BytesIO stream.
    تعيد None إذا كان النص فارغًا.
    """
    if not text_to_encode:
        return None
    
    qr_img = qrcode.make(text_to_encode)
    img_byte_arr = io.BytesIO()
    qr_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0) # مهم لإعادة المؤشر إلى بداية الـ stream
    return img_byte_arr

async def qr_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    يعالج الأمر /qr لإنشاء وإرسال صورة QR Code.
    """
    if not context.args:
        await update.message.reply_text(
            "لاستخدام الأمر، أرسل: /qr <النص أو الرابط الذي تريد تحويله إلى QR Code>"
        )
        logger.info(f"User {update.effective_user.id} used /qr without arguments.")
        return

    text_to_encode = " ".join(context.args)
    logger.info(f"User {update.effective_user.id} requested QR for text: '{text_to_encode}'")

    qr_image_stream = await generate_qr_image(text_to_encode)

    if qr_image_stream:
        await update.message.reply_photo(photo=qr_image_stream, caption=f"QR Code لـ: {text_to_encode}")
        logger.info(f"Sent QR code for '{text_to_encode}' to user {update.effective_user.id}.")
    else:
        await update.message.reply_text("حدث خطأ أثناء إنشاء QR Code. يرجى التأكد من إدخال نص.")
        logger.error(f"Failed to generate QR code for user {update.effective_user.id} (empty text somehow passed checks).")


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يتعامل مع الأوامر غير المعروفة."""
    await context.bot.send_message(chat_id=update.effective_chat.id, text="عذراً، لم أفهم هذا الأمر.")
    logger.info(f"User {update.effective_user.id} sent an unknown command: {update.message.text}")

# --- الدالة الرئيسية لتشغيل البوت ---

if __name__ == '__main__':
    logger.info("Starting bot with QR Code feature...")

    # بناء التطبيق باستخدام التوكن
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # 1. إضافة معالج أمر /start
    application.add_handler(CommandHandler('start', start_command))

    # 2. إضافة معالج أمر /admin_test (تجريبي للمعرفات المضافة)
    application.add_handler(CommandHandler('admin_test', admin_test_command))
    
    # 3. إضافة معالج أمر /qr الجديد
    application.add_handler(CommandHandler('qr', qr_command_handler))

    # 4. إضافة معالج الرسائل النصية (الترديد)
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo_message))

    # 5. إضافة معالج للأوامر غير المعروفة
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    logger.info("Bot is now polling for updates...")
    application.run_polling()

    logger.info("Bot has stopped.")