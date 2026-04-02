import fitz
import openai
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DatasheetParser:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def parse_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text content from raw PDF bytes."""
        text = ""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            return text

    def extract_sensor_config(self, sensor_name: str, sensor_id: str, datasheet_text: str) -> Dict[str, Any]:
        """Use OpenAI to determine normal operating boundaries for a sensor from its datasheet."""
        prompt = f"""
You are an expert industrial engineer. Read the following datasheet text for the sensor '{sensor_name}' (ID: '{sensor_id}').
Extract the typical normal operating mean (mu) and standard deviation/tolerance (sigma). 
If the exact values are not explicitly stated, estimate realistic baseline operating conditions based on the text. 
Return EXACTLY valid JSON in the following format:
{{
  "sensor_id": "{sensor_id}",
  "sensor_name": "{sensor_name}",
  "mu": <number>,
  "sigma": <number>
}}

Datasheet extract:
{datasheet_text[:4000]}
"""
        try:
            res = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" },
                temperature=0.1
            )
            data = json.loads(res.choices[0].message.content)
            # Guarantee correct types
            return {
                "sensor_id": str(data.get("sensor_id", sensor_id)),
                "sensor_name": str(data.get("sensor_name", sensor_name)),
                 # Fallback to realistic float values if parsing fails internally
                "mu": float(data.get("mu", 50.0)),
                "sigma": float(data.get("sigma", 5.0))
            }
        except Exception as e:
            logger.error(f"Error extracting config via LLM: {e}")
            return {"sensor_id": sensor_id, "sensor_name": sensor_name, "mu": 50.0, "sigma": 5.0} # Fallback
