import requests
import json
import time

base = "http://127.0.0.1:8000"

print("--- HEALTH ---")
try:
    print(requests.get(f"{base}/health/status").json())
    print("Fetched sensors:", len(requests.get(f"{base}/health/sensors").json().get("sensors", {})))
except Exception as e:
    print("Health endpoints error:", e)

print("\n--- DETECT NORMAL ---")
payload = {"temperature": 180.0, "motor_current": 4.5, "vibration": 0.8, "speed": 160.0, "pressure": 4.5}
print(requests.post(f"{base}/anomaly/detect", json=payload).json())

print("\n--- DETECT FAULT (3x to trigger Orchestrator) ---")
payload_fault = {"temperature": 181.0, "motor_current": 7.5, "vibration": 2.8, "speed": 120.0, "pressure": 4.2}
print("1st:", requests.post(f"{base}/anomaly/detect", json=payload_fault).json())
print("2nd:", requests.post(f"{base}/anomaly/detect", json=payload_fault).json())
print("3rd (Triggers Orchestrator! This takes a few seconds...):")
# The 3rd one hits the threshold and should kick off the agents synchronously in our current design
res = requests.post(f"{base}/anomaly/detect", json=payload_fault)
print("3rd Result:", res.json())

print("\n--- HISTORY ---")
hist = requests.get(f"{base}/anomaly/history?limit=1").json()
events = hist.get("events", [])
if events:
    print(f"Latest Event Score: {events[0].get('anomaly_score')}")
    print(f"Latest Agent Advice snippet: {events[0].get('agent_advice', '')[:100]}...")
else:
    print("No events found in history.")
