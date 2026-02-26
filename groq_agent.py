"""
groq_agent.py
Core AI agent using Groq API (llama-3.3-70b-versatile) for:
  1. Analyzing prescriptions
  2. Recommending medications by disease
"""

import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are CardioAI, an expert cardiac medical AI assistant trained on cardiology knowledge.

Your role is to:
1. Analyze cardiac-related prescriptions and medical notes
2. Identify heart conditions and explain them clearly to patients
3. Provide detailed medication recommendations with dosages
4. Explain mechanisms, side effects, and monitoring requirements
5. Give lifestyle advice

STRICT RULES:
- Only address cardiac/cardiovascular conditions
- Always include a medical disclaimer
- Format your response as valid JSON only — no markdown, no extra text
- Be medically accurate, cite evidence-based guidelines (ACC/AHA, ESC)
- Use clear, patient-friendly language for explanations
- For emergency conditions (STEMI, hypertensive crisis), flag them prominently

RESPONSE FORMAT for prescription analysis:
{
  "emergency": false,
  "emergency_message": null,
  "conditions_identified": ["list of conditions"],
  "condition_details": [
    {
      "name": "Condition Name",
      "explanation": "Clear patient-friendly explanation",
      "severity": "mild|moderate|severe",
      "classification": "optional subtype info"
    }
  ],
  "medication_review": [
    {
      "name": "Drug Name",
      "dosage": "Xmg frequency",
      "drug_class": "Class name",
      "purpose": "Why it's prescribed",
      "mechanism": "How it works simply",
      "side_effects": ["list"],
      "monitoring": "What to watch for"
    }
  ],
  "drug_interactions": ["any interactions found"],
  "lifestyle_advice": ["list of advice"],
  "follow_up": "Follow-up recommendation",
  "summary": "Overall 2-3 sentence plain English summary",
  "disclaimer": "This analysis is for educational purposes only. Always follow your doctor's advice."
}

RESPONSE FORMAT for disease-based recommendation:
{
  "emergency": false,
  "emergency_message": null,
  "condition": {
    "name": "Full condition name",
    "explanation": "Comprehensive patient-friendly explanation",
    "causes": ["list of common causes"],
    "symptoms": ["list of symptoms"],
    "severity": "mild|moderate|severe",
    "classification": {}
  },
  "recommended_medications": [
    {
      "name": "Drug Name",
      "drug_class": "Class name",
      "dosage": "Typical dose range",
      "frequency": "How often",
      "mechanism": "Simple explanation",
      "side_effects": ["list"],
      "monitoring": "What to watch",
      "priority": "first-line|second-line|adjunct"
    }
  ],
  "lifestyle_advice": ["list"],
  "treatment_goals": ["target values / endpoints"],
  "when_to_seek_emergency": ["red flag symptoms"],
  "summary": "2-3 sentence overview",
  "disclaimer": "This information is for educational purposes only. Consult a licensed cardiologist for treatment."
}"""


class GroqCardiacAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment variables")
        self.client = Groq(api_key=api_key)
        self.model = GROQ_MODEL

    def _call(self, user_message: str, temperature: float = 0.2) -> dict:
        """Call Groq API and parse JSON response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {"error": "Failed to parse AI response", "raw": raw}
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    def analyze_prescription(self, prescription_data: dict) -> dict:
        """Analyze a parsed prescription"""
        raw_text = prescription_data.get("raw_text", "")
        medications = prescription_data.get("medications", [])
        conditions = prescription_data.get("conditions", [])
        patient_info = prescription_data.get("patient_info", {})

        prompt = f"""Analyze this cardiac prescription and provide a detailed medical analysis.

PRESCRIPTION TEXT:
{raw_text[:1500]}

EXTRACTED DATA:
- Patient Info: {json.dumps(patient_info)}
- Detected Conditions: {json.dumps(conditions)}
- Detected Medications: {json.dumps(medications)}

Please analyze this prescription comprehensively. Identify all cardiac conditions, review all medications, check for drug interactions, and provide clear patient-friendly explanations. Return valid JSON only."""

        result = self._call(prompt)
        result["source"] = "groq_ai"
        result["extracted"] = {
            "conditions": conditions,
            "medications": medications,
            "patient_info": patient_info,
        }
        return result

    def recommend_by_disease(self, condition: str, patient_info: dict = None) -> dict:
        """Get medication recommendations for a disease"""
        patient_context = ""
        if patient_info:
            patient_context = f"\nPatient context: {json.dumps(patient_info)}"
            if patient_info.get("age", 0) > 75:
                patient_context += "\n(Elderly patient - consider dose adjustments)"
            comorbidities = patient_info.get("comorbidities", [])
            if comorbidities:
                patient_context += f"\nComorbidities: {', '.join(comorbidities)}"

        prompt = f"""Provide comprehensive cardiac medication recommendations for: {condition}
{patient_context}

Include:
- Full condition explanation in patient-friendly language
- Evidence-based medication recommendations (ACC/AHA/ESC guidelines)
- Dosages, drug classes, mechanisms, side effects
- Lifestyle modifications
- Treatment targets and goals
- Emergency warning signs

Return valid JSON only."""

        result = self._call(prompt)
        result["source"] = "groq_ai"
        if patient_info:
            result["patient_context"] = patient_info
        return result

    def quick_explain(self, term: str) -> dict:
        """Quick explanation of a cardiac term or drug"""
        prompt = f"""Briefly explain this cardiac medical term or drug in patient-friendly language: "{term}"
Include: what it is, why it matters, and one key fact.
Return JSON with fields: term, explanation, key_fact, category."""
        return self._call(prompt, temperature=0.3)
