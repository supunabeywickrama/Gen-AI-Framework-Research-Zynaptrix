import requests
import json

url = "http://localhost:8000/api/copilot/invoke"
payload = {
    "machine_id": "PUMP-001",
    "machine_state": "PUMP_FAULT",
    "anomaly_id": 1,
    "user_query": "How do I fix this leak?",
    "suspect_sensor": "Vibration",
    "recent_readings": {"temperature": 85.5}
}

print(f"📡 Sending mock inquiry for anomaly #1...")
try:
    response = requests.post(url, json=payload, timeout=10)
    print(f"✅ Response Status: {response.status_code}")
    print(f"📄 Response Body: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")
