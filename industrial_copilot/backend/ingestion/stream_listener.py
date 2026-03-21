"""
stream_listener.py — A background daemon polling InfluxDB and pushing live streams to FastAPI inference.
"""
import time
import requests
import logging
import sys
import os

# Allow running from working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from influxdb_client import InfluxDBClient
from config.influx_config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, INFLUX_MEASUREMENT
from config.settings import SENSOR_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def poll_and_forward():
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()
    
    # Query to get the last row
    query = f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: -5s)
          |> filter(fn: (r) => r["_measurement"] == "{INFLUX_MEASUREMENT}")
          |> last()
    '''
    
    last_time = None
    
    logger.info("📡 Stream Listener started. Polling InfluxDB every 1s...")
    while True:
        try:
            tables = query_api.query(query, org=INFLUX_ORG)
            reading = {}
            current_time = None
            
            for table in tables:
                for record in table.records:
                    field = record.get_field()
                    val = record.get_value()
                    reading[field] = val
                    current_time = record.get_time()
                    
            if reading and current_time != last_time:
                # Check if we have all columns
                if all(col in reading for col in SENSOR_COLUMNS):
                    # Forward to FastAPI
                    try:
                        api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
                        res = requests.post(f"{api_url}/anomaly/detect", json=reading, timeout=2)
                        if res.status_code == 200:
                            data = res.json()
                            logger.info(f"Forwarded reading. Anomaly: {data.get('is_anomaly')}, Score: {data.get('score'):.3f}")
                        else:
                            logger.warning(f"FastAPI returned status code: {res.status_code}")
                        last_time = current_time
                    except requests.exceptions.ConnectionError:
                        logger.error("FastAPI backend is offline. Retrying...")
                    
        except Exception as e:
            logger.error(f"Error polling InfluxDB: {e}")
            
        time.sleep(1.0)

if __name__ == "__main__":
    poll_and_forward()
