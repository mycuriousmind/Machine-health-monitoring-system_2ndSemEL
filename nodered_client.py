"""
nodered_client.py — Node-RED HTTP Client Module
================================================
Lightweight Python client to push telemetry and alerts to Node-RED's
HTTP endpoints. Node-RED then handles routing to InfluxDB, Back4App,
and ThingSpeak.

Uses only stdlib (urllib) — no extra dependencies.

Capabilities:
  - push_telemetry()   : send sensor readings to Node-RED /api/telemetry
  - push_alert()       : send alert events to Node-RED /api/alert
  - test_connection()  : verify Node-RED is running and reachable
"""

import os
import json
import time
import urllib.request
import urllib.error

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "integration_config.json")


class NodeREDClient:
    """
    HTTP client for pushing telemetry and alerts to Node-RED endpoints.
    """

    def __init__(self):
        self.config = self._load_config()
        self.enabled = self.config.get("enabled", False)
        self.base_url = self.config.get("url", "http://localhost:1880")
        self.telemetry_endpoint = self.config.get("telemetry_endpoint", "/api/telemetry")
        self.alert_endpoint = self.config.get("alert_endpoint", "/api/alert")
        self.last_status = "Not Connected"
        self.last_error = ""
        self._last_push_time = 0.0
        self._min_push_interval = 2  # seconds between telemetry pushes

    def _load_config(self):
        """Load Node-RED config from integration_config.json"""
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    full_config = json.load(f)
                return full_config.get("nodered", {})
            except Exception:
                pass
        return {"enabled": False}

    def _post(self, endpoint, data):
        """
        Send an HTTP POST request to a Node-RED endpoint.

        Args:
            endpoint : str — e.g., '/api/telemetry'
            data     : dict — payload to send as JSON

        Returns:
            True on success (2xx response), False on failure.
        """
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5) as response:
                # Any 2xx is success
                if 200 <= response.status < 300:
                    self.last_status = "Connected"
                    self.last_error = ""
                    return True
                else:
                    self.last_status = f"HTTP {response.status}"
                    return False
        except urllib.error.URLError as e:
            self.last_status = "Connection Failed"
            self.last_error = str(e.reason)
            return False
        except Exception as e:
            self.last_status = "Error"
            self.last_error = str(e)
            return False

    def test_connection(self):
        """
        Test if Node-RED is running by checking the root URL.
        Returns True if reachable, False otherwise.
        """
        if not self.enabled:
            self.last_status = "Disabled"
            return False

        try:
            req = urllib.request.Request(self.base_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                if 200 <= response.status < 400:
                    self.last_status = "Connected"
                    self.last_error = ""
                    return True
        except Exception as e:
            self.last_status = "Not Reachable"
            self.last_error = str(e)
        return False

    def push_telemetry(self, temp, current, vibration, rpm, battery,
                       fused_prob, cnn_prob, rf_prob, machine_id="esp32_node_01"):
        """
        Push telemetry data to Node-RED's /api/telemetry endpoint.

        Rate-limited to self._min_push_interval seconds.

        Args:
            temp        : float — temperature in °C
            current     : float — current in Amps
            vibration   : float — vibration RMS
            rpm         : float — rotational speed
            battery     : float — battery percentage
            fused_prob  : float — fused fault probability [0, 1]
            cnn_prob    : float — CNN fault probability [0, 1]
            rf_prob     : float — RF fault probability [0, 1]
            machine_id  : str — machine identifier

        Returns:
            True on success, False on failure or rate-limit skip.
        """
        if not self.enabled:
            return False

        # Rate limit
        now = time.time()
        if now - self._last_push_time < self._min_push_interval:
            return False

        data = {
            "temperature": round(float(temp), 2),
            "current": round(float(current), 3),
            "vibration": round(float(vibration), 4),
            "rpm": round(float(rpm), 1),
            "battery": round(float(battery), 1),
            "fused_prob": round(float(fused_prob), 4),
            "cnn_prob": round(float(cnn_prob), 4),
            "rf_prob": round(float(rf_prob), 4),
            "machine_id": machine_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        success = self._post(self.telemetry_endpoint, data)
        if success:
            self._last_push_time = now
        return success

    def push_alert(self, alert_type, message, severity="WARNING",
                   machine_id="esp32_node_01"):
        """
        Push an alert event to Node-RED's /api/alert endpoint.

        Args:
            alert_type : str — e.g., 'FAULT_DETECTED', 'MAINTENANCE_DUE'
            message    : str — human-readable alert message
            severity   : str — 'INFO', 'WARNING', 'CRITICAL'
            machine_id : str — machine identifier

        Returns:
            True on success, False on failure.
        """
        if not self.enabled:
            return False

        data = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "machine_id": machine_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        return self._post(self.alert_endpoint, data)
