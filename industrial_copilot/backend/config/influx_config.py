"""
influx_config.py — InfluxDB 2.x connection configuration.
Reads credentials from .env file via environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

INFLUX_URL    = os.getenv("INFLUX_URL",    "http://localhost:8086")
INFLUX_TOKEN  = os.getenv("INFLUX_API_KEY",  "YOUR_INFLUXDB_TOKEN")
INFLUX_ORG    = os.getenv("INFLUX_ORG",    "factory")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "machine_sensors")

INFLUX_MEASUREMENT = "machine_sensor"
