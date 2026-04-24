from flask import Flask, redirect, url_for, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import os
import numpy as np
import logging
import easyocr
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import warnings
from PIL import Image
import io
import uuid

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global session state (use session in production)
state = {
    'path_to_upload': None,
    'global_text': "",
    'lang': None,
    'detected_language': None,
    'char_count': 0,
    'word_count': 0,
    'confidence_score': 0.0
}

# Load detection model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detector.keras')

try:
    import tensorflow_hub as hub
    custom_objects = {'KerasLayer': hub.KerasLayer}
    detector_model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️  Could not load model with hub: {e}")
    try:
        detector_model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded (without hub)")
    except Exception as e2:
        print(f"❌ Model load failed: {e2}")
        detector_model = None

LANGUAGE_NAMES = {0: "English", 1: "Hindi", 2: "Telugu"}
LANGUAGE_CODES = {"English": "en", "Hindi": "hi", "Telugu": "te"}
LANGUAGE_FLAGS = {"English": "🇬🇧", "Hindi": "🇮🇳", "Telugu": "🇮🇳"}
LANGUAGE_SCRIPTS = {"English": "Latin Script", "Hindi": "Devanagari Script", "Telugu": "Telugu Script"}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_text(text, line_length=80):
    """Format text into readable lines."""
    lines = []
    current_line = ""
    for word in text.split():
        if len(current_line) + len(word) + 1 <= line_length:
            current_line = (current_line + " " + word).strip()
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


@app.route('/', methods=['GET'])
def index():
    return render_template("intro.html")


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload and language detection."""
    try:
        if 'file' not in request.files:
            return render_template("error.html",
                                   error_title="No File Selected",
                                   error_msg="Please select an image file to upload.")

        f = request.files['file']
        if f.filename == '':
            return render_template("error.html",
                                   error_title="Empty Filename",
                                   error_msg="Please choose a valid image file.")

        if not allowed_file(f.filename):
            return render_template("error.html",
                                   error_title="Invalid File Type",
                                   error_msg=f"Supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WEBP")

        # Save with unique name to avoid collisions
        ext = f.filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_name)
        f.save(file_path)
        state['path_to_upload'] = file_path

        # Preprocess image
        img = Image.open(file_path)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if 'A' in img.mode:
                background.paste(img, mask=img.split()[-1])
            else:
                background.paste(img)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(img)
        resize = tf.image.resize(img_array, (256, 256)) / 255.0
        img_input = tf.expand_dims(resize, 0)

        if detector_model is None:
            # Fallback: guess English
            detected_name = "English"
            confidence = 0.70
        else:
            predictions = detector_model.predict(img_input, verbose=0)
            predicted_idx = int(np.argmax(predictions, axis=1)[0])
            confidence = float(np.max(predictions))
            detected_name = LANGUAGE_NAMES.get(predicted_idx, "English")

        state['lang'] = LANGUAGE_CODES.get(detected_name, 'en')
        state['detected_language'] = detected_name
        state['confidence_score'] = round(confidence * 100, 1)

        return render_template("detect.html",
                               language=detected_name,
                               flag=LANGUAGE_FLAGS.get(detected_name, "🌐"),
                               script=LANGUAGE_SCRIPTS.get(detected_name, "Unknown"),
                               confidence=state['confidence_score'])

    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return render_template("error.html",
                               error_title="Processing Error",
                               error_msg=f"Could not process image: {str(e)}")


@app.route('/extract', methods=['GET', 'POST'])
def extract():
    """Run OCR and show extracted text."""
    try:
        lang = state.get('lang')
        path = state.get('path_to_upload')

        if not lang or not path:
            return redirect(url_for('index'))

        logging.getLogger().setLevel(logging.ERROR)
        reader = easyocr.Reader([lang], verbose=False)
        results = reader.readtext(path)

        extracted = ""
        total_conf = 0.0
        for (bbox, text, prob) in results:
            extracted += " " + text
            total_conf += prob

        extracted = extracted.strip()
        state['global_text'] = extracted
        state['char_count'] = len(extracted)
        state['word_count'] = len(extracted.split())
        avg_conf = round((total_conf / len(results) * 100), 1) if results else 0

        return render_template("extracted.html",
                               text=extracted,
                               language=state['detected_language'],
                               lang_code=lang.upper(),
                               char_count=state['char_count'],
                               word_count=state['word_count'],
                               ocr_confidence=avg_conf)

    except Exception as e:
        app.logger.error(f"OCR error: {str(e)}")
        return render_template("error.html",
                               error_title="OCR Failed",
                               error_msg=f"Text extraction failed: {str(e)}")


@app.route('/format_normal', methods=['POST'])
def format_normal():
    """Simple formatting without AI model."""
    text = state.get('global_text', '')
    state['global_text'] = format_text(text)
    return render_template("download.html",
                           text=state['global_text'],
                           language=state.get('detected_language', 'Unknown'))


@app.route('/format_advanced', methods=['POST'])
def format_advanced():
    """Advanced formatting - spellcheck for English."""
    text = state.get('global_text', '')
    lang = state.get('lang', 'en')

    if lang == 'en':
        try:
            from spellchecker import SpellChecker
            spell = SpellChecker()
            words = text.split()
            corrected = []
            for word in words:
                clean = ''.join(c for c in word if c.isalpha())
                if clean and clean.lower() in spell.unknown([clean.lower()]):
                    correction = spell.correction(clean.lower())
                    if correction:
                        corrected.append(correction)
                    else:
                        corrected.append(word)
                else:
                    corrected.append(word)
            text = ' '.join(corrected)
        except Exception as e:
            app.logger.warning(f"Spell check skipped: {e}")

    state['global_text'] = format_text(text)
    return render_template("download.html",
                           text=state['global_text'],
                           language=state.get('detected_language', 'Unknown'))


@app.route('/download/pdf', methods=['POST'])
def download_pdf():
    """Generate and download PDF."""
    text = state.get('global_text', '')
    lang = state.get('detected_language', 'Document')

    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=20,
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=18,
        textColor=colors.HexColor('#333333'),
    )

    story = [
        Paragraph(f"Extracted Text — {lang}", title_style),
        Spacer(1, 0.2 * inch),
        Paragraph(text.replace('\n', '<br/>'), body_style),
    ]
    doc.build(story)
    output.seek(0)
    return send_file(output, mimetype='application/pdf',
                     as_attachment=True, download_name='extracted_text.pdf')


@app.route('/download/txt', methods=['POST'])
def download_txt():
    """Download as plain text file."""
    text = state.get('global_text', '')
    output = io.BytesIO(text.encode('utf-8'))
    return send_file(output, mimetype='text/plain',
                     as_attachment=True, download_name='extracted_text.txt')


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'model_loaded': detector_model is not None,
        'supported_languages': ['English', 'Hindi', 'Telugu']
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
