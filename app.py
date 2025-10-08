import os
import io
import requests
from flask import Flask, request, jsonify, render_template
import pdfplumber
from dotenv import load_dotenv

# Load environment variables from .env (for local testing)
load_dotenv()

app = Flask(__name__)

# Hugging Face API Key from environment variable
HF_API_KEY = os.getenv("HF_API_KEY")  # Make sure to set this in .env or Render
if not HF_API_KEY:
    raise ValueError("Hugging Face API key not found. Set HF_API_KEY in .env or Render environment variables.")

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def extract_text_from_pdf(file_stream):
    try:
        with pdfplumber.open(file_stream) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
                if len(text) > 8000:  # limit for API
                    break
        return text[:8000]
    except Exception as e:
        print("PDF Error:", e)
        return None

def summarize_text(text):
    payload = {"inputs": text, "parameters": {"max_length": 300, "min_length": 100}}
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code != 200:
            return [f"Error from Hugging Face: {response.text}"]
        result = response.json()
        summary_text = result[0].get("summary_text", "")
        bullets = ["• " + s.strip() for s in summary_text.split(". ") if s.strip()]
        return bullets
    except Exception as e:
        return [f"API Error: {str(e)}"]

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/summarize', methods=['POST'])
def summarize_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['pdf_file']
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Only PDF files allowed"}), 400

    text = extract_text_from_pdf(io.BytesIO(file.read()))
    if not text or len(text.strip()) < 50:
        return jsonify({"status": "error", "message": "Could not extract enough text from PDF"}), 400

    summary = summarize_text(text)
    return jsonify({"status": "success", "summary": summary}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7000))
    app.run(debug=True, host="0.0.0.0", port=port)
