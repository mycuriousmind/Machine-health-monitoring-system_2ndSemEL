"""
influxdb_manager.py — InfluxDB 2.x Integration Module
=====================================================
Provides write and query operations against InfluxDB 2.x for the
Machine Health Monitoring System. Uses the official influxdb-client library.

Capabilities:
  - write_telemetry()   : write a single telemetry point
  - query_recent()      : query recent telemetry (last N minutes)
  - query_health_history() : query fault probability over time
  - get_statistics()    : min/max/mean stats for each field
  - test_connection()   : verify InfluxDB is reachable and configured
"""

import os
import json
import time

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "integration_config.json")

# ── Graceful import ──────────────────────────────────────────────────────────
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUX_AVAILABLE = True
except ImportError:
    INFLUX_AVAILABLE = False
    print("[influxdb_manager] influxdb-client not installed. Run: pip install influxdb-client")


class InfluxDBManager:
    """
    Manages InfluxDB 2.x connections and operations for the Machine Health
    Monitoring System.
    """

    def __init__(self):
        self.config = self._load_config()
        self.enabled = self.config.get("enabled", False)
        self.last_write_time = 0.0
        self.min_write_interval = 5  # seconds between writes
        self.last_status = "Not Connected"
        self.last_error = ""
        self._client = None
        self._write_api = None
        self._query_api = None

        if self.enabled and INFLUX_AVAILABLE:
            self._connect()

    def _load_config(self):
        """Load InfluxDB config from integration_config.json"""
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    full_config = json.load(f)
                return full_config.get("influxdb", {})
            except Exception:
                pass
        return {"enabled": False}

    def _connect(self):
        """Establish connection to InfluxDB."""
        try:
            url = self.config.get("url", "http://localhost:8086")
            token = self.config.get("token", "")
            org = self.config.get("org", "machine-health")

            self._client = InfluxDBClient(url=url, token=token, org=org)
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            self._query_api = self._client.query_api()
            self.last_status = "Connected"
            self.last_error = ""
        except Exception as e:
            self.last_status = "Connection Error"
            self.last_error = str(e)
            self._client = None

    def test_connection(self):
        """
        Test if InfluxDB is reachable and the bucket exists.
        Returns True on success, False on failure.
        """
        if not INFLUX_AVAILABLE:
            self.last_status = "Library Missing"
            self.last_error = "influxdb-client not installed"
            return False

        if not self.enabled:
            self.last_status = "Disabled"
            return False

        try:
            if self._client is None:
                self._connect()
            health = self._client.health()
            if health.status == "pass":
                self.last_status = "Connected"
                self.last_error = ""
                return True
            else:
                self.last_status = "Unhealthy"
                self.last_error = health.message or "Unknown health issue"
                return False
        except Exception as e:
            self.last_status = "Connection Error"
            self.last_error = str(e)
            return False

    def write_telemetry(self, temp, current, vibration, rpm, battery,
                        fused_prob, cnn_prob, rf_prob, machine_id="esp32_node_01"):
        """
        Write a single telemetry data point to InfluxDB.

        Rate-limited to self.min_write_interval seconds.
        Returns True on success, False on failure or rate-limit skip.
        """
        if not self.enabled or not INFLUX_AVAILABLE or self._write_api is None:
            return False

        # Rate limiting
        now = time.time()
        if now - self.last_write_time < self.min_write_interval:
            return False

        try:
            bucket = self.config.get("bucket", "telemetry")

            # Determine status string for tagging
            status = "FAULTY" if fused_prob >= 0.5 else "HEALTHY"

            point = (
                Point("machine_telemetry")
                .tag("machine_id", machine_id)
                .tag("status", status)
                .field("temperature", float(temp))
                .field("current", float(current))
                .field("vibration", float(vibration))
                .field("rpm", float(rpm))
                .field("battery", float(battery))
                .field("fused_prob", float(fused_prob))
                .field("cnn_prob", float(cnn_prob))
                .field("rf_prob", float(rf_prob))
            )

            self._write_api.write(bucket=bucket, record=point)
            self.last_write_time = now
            self.last_status = "Connected & Writing"
            self.last_error = ""
            return True

        except Exception as e:
            self.last_status = "Write Error"
            self.last_error = str(e)
            return False

    def query_recent(self, minutes=60):
        """
        Query the most recent telemetry data points.

        Args:
            minutes: How far back to query (default 60 minutes)

        Returns:
            List of dicts with telemetry fields, or empty list on failure.
        """
        if not self.enabled or not INFLUX_AVAILABLE or self._query_api is None:
            return []

        try:
            bucket = self.config.get("bucket", "telemetry")
            query = f'''
                from(bucket: "{bucket}")
                    |> range(start: -{minutes}m)
                    |> filter(fn: (r) => r._measurement == "machine_telemetry")
                    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
                    |> sort(columns: ["_time"], desc: false)
                    |> limit(n: 500)
            '''

            result = self._query_api.query(query)
            records = []
            for table in result:
                for record in table.records:
                    records.append({
                        "time": record.get_time().isoformat(),
                        "temperature": record.values.get("temperature", 0),
                        "current": record.values.get("current", 0),
                        "vibration": record.values.get("vibration", 0),
                        "rpm": record.values.get("rpm", 0),
                        "battery": record.values.get("battery", 100),
                        "fused_prob": record.values.get("fused_prob", 0),
                        "cnn_prob": record.values.get("cnn_prob", 0),
                        "rf_prob": record.values.get("rf_prob", 0),
                        "status": record.values.get("status", "UNKNOWN"),
                    })
            return records

        except Exception as e:
            self.last_error = str(e)
            return []

    def query_health_history(self, hours=24):
        """
        Query fault probability history for the last N hours.

        Returns:
            List of dicts with time, fused_prob, cnn_prob, rf_prob.
        """
        if not self.enabled or not INFLUX_AVAILABLE or self._query_api is None:
            return []

        try:
            bucket = self.config.get("bucket", "telemetry")
            query = f'''
                from(bucket: "{bucket}")
                    |> range(start: -{hours}h)
                    |> filter(fn: (r) => r._measurement == "machine_telemetry")
                    |> filter(fn: (r) => r._field == "fused_prob" or
                                         r._field == "cnn_prob" or
                                         r._field == "rf_prob")
                    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
                    |> sort(columns: ["_time"], desc: false)
            '''

            result = self._query_api.query(query)
            records = []
            for table in result:
                for record in table.records:
                    records.append({
                        "time": record.get_time().isoformat(),
                        "fused_prob": record.values.get("fused_prob", 0),
                        "cnn_prob": record.values.get("cnn_prob", 0),
                        "rf_prob": record.values.get("rf_prob", 0),
                    })
            return records

        except Exception as e:
            self.last_error = str(e)
            return []

    def get_statistics(self, hours=24):
        """
        Get min/max/mean statistics for each telemetry field.

        Returns:
            Dict with field names as keys, each containing min, max, mean.
        """
        if not self.enabled or not INFLUX_AVAILABLE or self._query_api is None:
            return {}

        try:
            bucket = self.config.get("bucket", "telemetry")
            fields = ["temperature", "current", "vibration", "rpm", "battery",
                       "fused_prob", "cnn_prob", "rf_prob"]
            stats = {}

            for field in fields:
                query = f'''
                    from(bucket: "{bucket}")
                        |> range(start: -{hours}h)
                        |> filter(fn: (r) => r._measurement == "machine_telemetry")
                        |> filter(fn: (r) => r._field == "{field}")
                        |> reduce(
                            fn: (r, accumulator) => ({{
                                count: accumulator.count + 1.0,
                                total: accumulator.total + r._value,
                                min_val: if r._value < accumulator.min_val then r._value else accumulator.min_val,
                                max_val: if r._value > accumulator.max_val then r._value else accumulator.max_val
                            }}),
                            identity: {{count: 0.0, total: 0.0, min_val: 999999.0, max_val: -999999.0}}
                        )
                '''
                result = self._query_api.query(query)
                for table in result:
                    for record in table.records:
                        count = record.values.get("count", 0)
                        total = record.values.get("total", 0)
                        stats[field] = {
                            "min": record.values.get("min_val", 0),
                            "max": record.values.get("max_val", 0),
                            "mean": total / count if count > 0 else 0,
                            "count": int(count)
                        }

            return stats

        except Exception as e:
            self.last_error = str(e)
            return {}

    def close(self):
        """Close the InfluxDB client connection."""
        if self._client:
            self._client.close()
