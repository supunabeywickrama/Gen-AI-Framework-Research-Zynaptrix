"""
streamlit_dashboard.py — Main UI for the Industrial AI Copilot
Decoupled for Phase 7: Passively monitors the API stream without local mock files.
"""
import streamlit as st
import requests
import time
import os

from dashboard.graphs import create_sensor_gauge, create_anomaly_score_chart

API_BASE = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Industrial AI Copilot", layout="wide", page_icon="🏭")

# --- State Initialization ---
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False
if "recent_scores" not in st.session_state:
    st.session_state.recent_scores = []
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.7187  # Fallback

# --- Fetch Config ---
@st.cache_data
def load_sensor_config():
    try:
        res = requests.get(f"{API_BASE}/health/sensors")
        return res.json().get("sensors", {})
    except:
        st.error("Could not connect to FastAPI backend. Is it running?")
        return {}

sensor_config = load_sensor_config()

# --- Layout ---
st.title("🏭 Plant Operations — Live Dashboard")

# Top controls
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("Passively monitoring continuous streams from **InfluxDB** via the **AI Copilot** background daemon.")
with col2:
    if st.button("▶️ Connect to Live Stream" if not st.session_state.is_listening else "⏸️ Disconnect Stream", use_container_width=True):
        st.session_state.is_listening = not st.session_state.is_listening
        st.rerun()

# Layout structure
left_pane, right_sidebar = st.columns([10, 4])

# Active components within the left pane
with left_pane:
    # Sensor Gauges Row
    st.markdown("### 📡 Live Sensor Feed")
    cols = st.columns(5)
    gauge_placeholders = [col.empty() for col in cols]
    
    # Anomaly Chart Row
    st.markdown("### 📈 Detection Model")
    chart_placeholder = st.empty()

# Active components within the right sidebar (AI Copilot Feed)
with right_sidebar:
    st.markdown("### 🤖 Orchestrator Logs")
    feed_placeholder = st.empty()

# --- Listening Loop ---
if st.session_state.is_listening:
    last_processed_time = time.time()
    
    while st.session_state.is_listening:
        try:
            # 1. Fetch Latest Streamed Point from Memory Cache
            res = requests.get(f"{API_BASE}/anomaly/latest").json()
            data = res.get("data", {})
            
            if not data or not data.get("reading"):
                # No data in cache yet
                left_pane.warning("⏳ Connected to API, but no live InfluxDB stream detected. Ensure `simulator` and `stream_listener` are running.")
                time.sleep(2)
                st.rerun()
                
            reading = data.get("reading")
            is_anomaly = data.get("is_anomaly", False)
            score = data.get("score", 0.0)
            threshold = data.get("threshold", st.session_state.threshold)
            
            # Update history buffer 
            st.session_state.threshold = threshold
            st.session_state.recent_scores.append(score)
            if len(st.session_state.recent_scores) > 100:
                st.session_state.recent_scores.pop(0)
                
            # 2. Update Gauges
            for i, (sensor_name, val) in enumerate(reading.items()):
                config = sensor_config.get(sensor_name, {})
                fig = create_sensor_gauge(val, sensor_name, config)
                gauge_placeholders[i].plotly_chart(fig, use_container_width=True, key=f"gauge_{sensor_name}_{time.time()}")
                
            # 3. Update Chart
            chart_fig = create_anomaly_score_chart(st.session_state.recent_scores, st.session_state.threshold)
            chart_placeholder.plotly_chart(chart_fig, use_container_width=True, key=f"chart_{time.time()}")
            
            # 4. Update Agent Feed Sidebar 
            try:
                hist = requests.get(f"{API_BASE}/anomaly/history?limit=3").json()
                events = hist.get("events", [])
                
                with feed_placeholder.container():
                    if not events:
                        st.info("Operating normally. No events to display.")
                    for event in events:
                        severity_color = "🔴" if float(event["anomaly_score"]) > threshold * 2 else "🟠"
                        with st.expander(f"{severity_color} {event['event_time'][:19]} | {event['machine_state']}", expanded=True):
                            if event.get("suspect_sensor"):
                                st.write(f"**Suspect Sensor**: `{event['suspect_sensor']}`")
                            st.write(f"**Score**: {event['anomaly_score']:.4f}")
                            st.markdown("---")
                            st.markdown(event.get("agent_advice", "No advice recorded."))
            except Exception as e:
                pass
            
        except requests.exceptions.ConnectionError:
            st.error("API Connection Failed! Ensure FastAPI is running.")
            st.session_state.is_listening = False
            time.sleep(1)
            st.rerun()
            
        # UI Throttle 
        time.sleep(0.5)

elif not st.session_state.is_listening:
    left_pane.info("Stream disconnected. Click 'Connect to Live Stream' to begin monitoring.")
