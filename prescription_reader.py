"""
prescription_reader.py
Handles reading prescriptions from images, PDFs, or plain text.
"""
import os
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'
import subprocess
import sys

try:
    pytesseract.get_tesseract_version()
except:
    subprocess.run(['pip', 'install', 'pytesseract'], check=True)
import os
import re
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytesseract
import shutil
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
import pdfplumber

logger = logging.getLogger(__name__)

# ── Drug & Condition Dictionaries ──────────────────────────────────────────────

CARDIAC_DRUGS = {
    "lisinopril","enalapril","ramipril","captopril","benazepril","perindopril",
    "losartan","valsartan","olmesartan","irbesartan","telmisartan","candesartan",
    "metoprolol","carvedilol","bisoprolol","atenolol","propranolol","nebivolol",
    "amlodipine","nifedipine","diltiazem","verapamil","felodipine",
    "furosemide","hydrochlorothiazide","spironolactone","torsemide","chlorthalidone",
    "atorvastatin","rosuvastatin","simvastatin","pravastatin","lovastatin",
    "aspirin","clopidogrel","ticagrelor","prasugrel","warfarin",
    "apixaban","rivaroxaban","dabigatran","edoxaban","enoxaparin",
    "nitroglycerin","isosorbide","digoxin","amiodarone","ivabradine",
    "sacubitril","dapagliflozin","empagliflozin","ezetimibe",
}

FREQUENCY_MAP = {
    r'\bOD\b|\bonce daily\b|\bq\.?d\.?\b': 'Once Daily',
    r'\bBD\b|\bBID\b|\btwice daily\b|\bb\.?i\.?d\.?\b': 'Twice Daily',
    r'\bTDS\b|\bTID\b|\bthrice daily\b|\bt\.?i\.?d\.?\b': 'Three Times Daily',
    r'\bQID\b|\bfour times\b': 'Four Times Daily',
    r'\bHS\b|\bat bedtime\b': 'At Bedtime',
    r'\bSOS\b|\bPRN\b|\bas needed\b': 'As Needed',
}

CARDIAC_CONDITIONS = {
    "hypertension": ["high blood pressure","htn","elevated bp","bp"],
    "heart failure": ["chf","congestive heart failure","hfref","hfpef","cardiac failure"],
    "atrial fibrillation": ["afib","a-fib","atrial fib","af"],
    "coronary artery disease": ["cad","angina","ischemic heart disease","ihd"],
    "myocardial infarction": ["mi","heart attack","stemi","nstemi","ami"],
    "arrhythmia": ["tachycardia","bradycardia","palpitation","irregular heartbeat"],
    "hyperlipidemia": ["high cholesterol","dyslipidemia","hypercholesterolemia"],
}


def preprocess_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        img = np.array(Image.open(img_path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def ocr_image(path: str) -> str:
    try:
        processed = preprocess_image(path)
        pil_img = Image.fromarray(processed)
        text = pytesseract.image_to_string(pil_img, config="--oem 3 --psm 6")
        return _clean(text)
    except Exception as e:
        logger.warning(f"Advanced OCR failed, trying basic: {e}")
        return _clean(pytesseract.image_to_string(Image.open(path)))


def ocr_pdf(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return _clean("\n".join(parts))


def _clean(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def extract_medications(text: str) -> list:
    found = []
    text_lower = text.lower()
    for drug in CARDIAC_DRUGS:
        if drug in text_lower:
            dose_pat = re.compile(
                rf'\b{re.escape(drug)}\b[\s\w]*?(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml))',
                re.IGNORECASE
            )
            m = dose_pat.search(text)
            dosage = m.group(1) if m else "Not specified"
            freq = "Not specified"
            for pat, label in FREQUENCY_MAP.items():
                if re.search(pat, text, re.IGNORECASE):
                    freq = label
                    break
            found.append({"name": drug.title(), "dosage": dosage, "frequency": freq})
    return found


def extract_conditions(text: str) -> list:
    found = []
    text_lower = text.lower()
    for cond, aliases in CARDIAC_CONDITIONS.items():
        if any(a in text_lower for a in [cond] + aliases):
            found.append(cond.title())
    return list(set(found))


def extract_patient_info(text: str) -> dict:
    info = {}
    age = re.search(r'\b(\d{1,3})\s*(?:year|yr|y)s?\s*(?:old|/|male|female|m|f)\b', text, re.I)
    if age:
        info["age"] = int(age.group(1))
    if re.search(r'\b(?:male|man|mr\.?)\b', text, re.I):
        info["gender"] = "Male"
    elif re.search(r'\b(?:female|woman|mrs\.?|ms\.?)\b', text, re.I):
        info["gender"] = "Female"
    bp = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?', text, re.I)
    if bp:
        info["blood_pressure"] = f"{bp.group(1)}/{bp.group(2)} mmHg"
    hr = re.search(r'(?:hr|heart rate|pulse)[:\s]*(\d{2,3})\s*(?:bpm)?', text, re.I)
    if hr:
        info["heart_rate"] = f"{hr.group(1)} bpm"
    return info


class PrescriptionReader:
    def read(self, source: str, source_type: str = "text") -> dict:
        if source_type == "image":
            raw = ocr_image(source)
        elif source_type == "pdf":
            raw = ocr_pdf(source)
        else:
            raw = str(source)
        return {
            "raw_text": raw,
            "medications": extract_medications(raw),
            "conditions": extract_conditions(raw),
            "patient_info": extract_patient_info(raw),
            "source_type": source_type,
        }

    def read_from_bytes(self, file_bytes: bytes, filename: str) -> dict:
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                return self.read(tmp_path, "image")
            elif suffix == ".pdf":
                return self.read(tmp_path, "pdf")
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
        finally:
            os.unlink(tmp_path)
