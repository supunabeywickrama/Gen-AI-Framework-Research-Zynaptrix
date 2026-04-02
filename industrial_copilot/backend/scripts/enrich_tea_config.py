"""
Script to enrich TEA_0001 sensor configs with realistic operating ranges and icon types.
Run from the backend directory: python scripts/enrich_tea_config.py
"""
import openai
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

sensors = [
    ("LEM Current Sensor",  "lem_1",       "current"),
    ("Thermistor",          "thermistor_1", "temperature"),
    ("Encoder",             "encoder_!",    "speed"),
    ("Ground Fault CT",     "ct_1",         "current"),
]

config_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "sensor_configs.json")
with open(config_path) as f:
    all_configs = json.load(f)

for sensor_name, sensor_id, icon_type in sensors:
    prompt = f"""You are an expert industrial instrumentation engineer.
Analyze sensor: "{sensor_name}" (ID: {sensor_id}) and return EXACTLY this JSON with realistic values:
{{
  "sensor_id": "{sensor_id}",
  "sensor_name": "{sensor_name}",
  "unit": "<SI or industry-standard unit>",
  "mu": <float: typical nominal operating value>,
  "sigma": <float: realistic std deviation during normal operation>,
  "min_normal": <float: lower boundary of healthy operating range>,
  "max_normal": <float: upper boundary of healthy operating range>,
  "fault_high": <float: definitive high-side fault/alarm threshold>,
  "fault_low": <float or null: definitive low-side fault (null if N/A)>,
  "fault_direction": "high or low or both",
  "icon_type": "{icon_type}"
}}"""

    try:
        res = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.05
        )
        data = json.loads(res.choices[0].message.content)

        # Merge with existing config, preserving any extra fields
        existing = all_configs.get("TEA_0001", {}).get(sensor_id, {})
        all_configs.setdefault("TEA_0001", {})[sensor_id] = {
            **existing,
            **data,
            "icon_type": icon_type,  # Enforce our icon type
        }
        print(f"✅ {sensor_name}: mu={data.get('mu')}, unit={data.get('unit')}, min={data.get('min_normal')}, max={data.get('max_normal')}")
    except Exception as e:
        print(f"❌ Failed for {sensor_name}: {e}")

with open(config_path, "w") as f:
    json.dump(all_configs, f, indent=2)

print("\n✅ sensor_configs.json updated successfully!")
print(json.dumps(all_configs.get("TEA_0001", {}), indent=2))
