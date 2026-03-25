"""
influx_writer.py — Clean abstraction for writing points to InfluxDB.
"""
import logging
from typing import Dict

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:
    pass

from config.influx_config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, INFLUX_MEASUREMENT

logger = logging.getLogger(__name__)

class InfluxWriter:
    """Wrapper for InfluxDB connections and writing points."""
    def __init__(self):
        self.client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = INFLUX_BUCKET
        self.org = INFLUX_ORG
        self.measurement = INFLUX_MEASUREMENT

    def write_sensor_reading(self, reading: Dict[str, any], state: str = "unknown"):
        """Writes a single sensor reading dict to InfluxDB."""
        machine_id = reading.get("machine_id", "unknown_machine")
        point = (
            Point(self.measurement)
            .tag("machine_id", machine_id)
            .tag("state", state)
            .field("temperature",   round(reading["temperature"], 3))
            .field("motor_current", round(reading["motor_current"], 3))
            .field("vibration",     round(reading["vibration"], 3))
            .field("speed",         round(reading["speed"], 3))
            .field("pressure",      round(reading["pressure"], 3))
        )
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        
    def close(self):
        self.client.close()
