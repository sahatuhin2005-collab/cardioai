"""
app.py - Flask REST API for Cardiac Medical AI Agent (Groq-powered)
"""

import os
import logging
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from prescription_reader import PrescriptionReader
from groq_agent import GroqCardiacAgent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".pdf"}

reader = PrescriptionReader()

try:
    agent = GroqCardiacAgent()
    logger.info("✅ Groq AI Agent initialized successfully")
except Exception as e:
    logger.error(f"❌ Groq Agent init failed: {e}")
    agent = None


def ok(data: dict, status: int = 200):
    return jsonify({"success": True, **data}), status


def err(msg: str, status: int = 400):
    return jsonify({"success": False, "error": msg}), status


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return ok({
        "status": "healthy",
        "groq_ready": agent is not None,
        "model": "llama-3.3-70b-versatile",
        "version": "2.0.0"
    })


@app.route("/api/read-prescription", methods=["POST"])
def read_prescription():
    """
    Accepts:
      - JSON: { "text": "..." }
      - Multipart: file field (image/PDF)
    """
    if agent is None:
        return err("AI Agent not initialized. Check GROQ_API_KEY.", 503)

    try:
        # Text input via JSON
        if request.is_json:
            data = request.get_json()
            text = data.get("text", "").strip()
            if not text:
                return err("No text provided")
            parsed = reader.read(text, "text")

        # File upload
        elif "file" in request.files:
            file = request.files["file"]
            if not file or not file.filename:
                return err("No file selected")
            filename = secure_filename(file.filename)
            if Path(filename).suffix.lower() not in ALLOWED_EXT:
                return err(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXT)}")
            parsed = reader.read_from_bytes(file.read(), filename)

        # Text input via form
        elif request.form.get("text"):
            parsed = reader.read(request.form["text"].strip(), "text")

        else:
            return err("Provide 'text' (JSON/form) or upload a 'file'")

        # Run Groq AI analysis
        analysis = agent.analyze_prescription(parsed)

        return ok({
            "parsed": parsed,
            "analysis": analysis,
        })

    except Exception as e:
        logger.error(f"read_prescription error: {e}", exc_info=True)
        return err(f"Processing failed: {str(e)}", 500)


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Body: { "condition": "hypertension", "patient_info": {...} }
    """
    if agent is None:
        return err("AI Agent not initialized. Check GROQ_API_KEY.", 503)

    try:
        data = request.get_json()
        if not data:
            return err("JSON body required")

        condition = data.get("condition", "").strip()
        if not condition:
            return err("'condition' field is required")

        patient_info = data.get("patient_info", {})
        result = agent.recommend_by_disease(condition, patient_info)

        return ok({"recommendation": result})

    except Exception as e:
        logger.error(f"recommend error: {e}", exc_info=True)
        return err(f"Recommendation failed: {str(e)}", 500)


@app.route("/api/explain", methods=["POST"])
def explain_term():
    """Quick explain a cardiac term: { "term": "ACE inhibitor" }"""
    if agent is None:
        return err("AI Agent not initialized.", 503)
    try:
        data = request.get_json()
        term = data.get("term", "").strip()
        if not term:
            return err("'term' field required")
        result = agent.quick_explain(term)
        return ok({"explanation": result})
    except Exception as e:
        return err(str(e), 500)


@app.route("/api/conditions", methods=["GET"])
def list_conditions():
    conditions = [
        {"id": "hypertension", "name": "Hypertension (High Blood Pressure)", "emergency": False},
        {"id": "heart failure", "name": "Heart Failure (CHF)", "emergency": False},
        {"id": "atrial fibrillation", "name": "Atrial Fibrillation (AFib)", "emergency": False},
        {"id": "coronary artery disease", "name": "Coronary Artery Disease / Angina", "emergency": False},
        {"id": "myocardial infarction", "name": "Myocardial Infarction (Heart Attack)", "emergency": True},
        {"id": "hyperlipidemia", "name": "Hyperlipidemia (High Cholesterol)", "emergency": False},
        {"id": "arrhythmia", "name": "Cardiac Arrhythmia", "emergency": False},
        {"id": "cardiomyopathy", "name": "Cardiomyopathy", "emergency": False},
        {"id": "aortic stenosis", "name": "Aortic Stenosis", "emergency": False},
        {"id": "hypertensive crisis", "name": "Hypertensive Crisis", "emergency": True},
    ]
    return ok({"conditions": conditions})


@app.errorhandler(413)
def too_large(e):
    return err("File too large (max 16MB)", 413)


@app.errorhandler(404)
def not_found(e):
    return err("Endpoint not found", 404)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
