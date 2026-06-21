"""
back4app_manager.py — Back4App (Parse Server) Integration Module
================================================================
Provides REST API operations against Back4App Parse Server for the
Machine Health Monitoring System. Uses only stdlib (urllib) — no
extra dependencies.

Capabilities:
  - log_machine_event()      : create a MachineEvent record (fault, alert, etc.)
  - log_maintenance()        : create a MaintenanceLog record
  - get_recent_events()      : query recent machine events
  - get_maintenance_history() : query maintenance log
  - test_connection()        : verify Back4App is reachable and configured
"""

import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "integration_config.json")


class Back4AppManager:
    """
    Manages Back4App (Parse Server) REST API operations for the Machine
    Health Monitoring System.
    """

    def __init__(self):
        self.config = self._load_config()
        self.enabled = self.config.get("enabled", False)
        self.last_status = "Not Connected"
        self.last_error = ""
        self._last_event_time = 0.0
        self._min_event_interval = 10  # seconds between events (avoid flooding)

    def _load_config(self):
        """Load Back4App config from integration_config.json"""
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    full_config = json.load(f)
                return full_config.get("back4app", {})
            except Exception:
                pass
        return {"enabled": False}

    def _get_headers(self):
        """Build the required Parse REST API headers."""
        return {
            "X-Parse-Application-Id": self.config.get("app_id", ""),
            "X-Parse-REST-API-Key": self.config.get("rest_api_key", ""),
            "Content-Type": "application/json"
        }

    def _get_base_url(self):
        """Get the Parse Server URL."""
        return self.config.get("server_url", "https://parseapi.back4app.com")

    def _make_request(self, method, endpoint, data=None, params=None):
        """
        Make an HTTP request to Back4App Parse REST API.

        Args:
            method   : HTTP method (GET, POST, PUT, DELETE)
            endpoint : API endpoint path (e.g., /classes/MachineEvent)
            data     : dict to send as JSON body (for POST/PUT)
            params   : dict of query parameters (for GET)

        Returns:
            dict: parsed JSON response, or None on failure
        """
        base_url = self._get_base_url()
        url = f"{base_url}{endpoint}"

        if params:
            query_string = urllib.parse.urlencode(params)
            url = f"{url}?{query_string}"

        headers = self._get_headers()
        body = json.dumps(data).encode("utf-8") if data else None

        try:
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=10) as response:
                resp_data = response.read().decode("utf-8")
                return json.loads(resp_data)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            self.last_error = f"HTTP {e.code}: {error_body}"
            self.last_status = "API Error"
            return None
        except Exception as e:
            self.last_error = str(e)
            self.last_status = "Network Error"
            return None

    def test_connection(self):
        """
        Test connectivity to Back4App by querying the health endpoint.
        Returns True on success, False on failure.
        """
        if not self.enabled:
            self.last_status = "Disabled"
            return False

        app_id = self.config.get("app_id", "")
        rest_key = self.config.get("rest_api_key", "")

        if not app_id or app_id == "YOUR_BACK4APP_APP_ID":
            self.last_status = "Not Configured"
            self.last_error = "App ID not set in integration_config.json"
            return False
        if not rest_key or rest_key == "YOUR_BACK4APP_REST_API_KEY":
            self.last_status = "Not Configured"
            self.last_error = "REST API Key not set in integration_config.json"
            return False

        # Try to query MachineEvent class (with limit=0 just to check auth)
        result = self._make_request("GET", "/classes/MachineEvent", params={"limit": "0"})
        if result is not None:
            self.last_status = "Connected"
            self.last_error = ""
            return True
        return False

    def log_machine_event(self, event_type, severity, description,
                          telemetry=None, ai_predictions=None,
                          machine_id="esp32_node_01"):
        """
        Create a MachineEvent record in Back4App.

        Args:
            event_type     : str — e.g., 'FAULT_DETECTED', 'TEMPERATURE_ALERT',
                             'OVERCURRENT_ALERT', 'LOW_BATTERY', 'MANUAL_ALERT'
            severity       : str — 'INFO', 'WARNING', 'CRITICAL'
            description    : str — human-readable description of the event
            telemetry      : dict — optional snapshot of sensor readings
            ai_predictions : dict — optional AI model probabilities
            machine_id     : str — machine identifier

        Returns:
            dict with 'objectId' and 'createdAt' on success, None on failure.
        """
        if not self.enabled:
            return None

        # Rate limit
        now = time.time()
        if now - self._last_event_time < self._min_event_interval:
            return None

        class_name = self.config.get("classes", {}).get("machine_event", "MachineEvent")
        endpoint = f"/classes/{class_name}"

        data = {
            "eventType": event_type,
            "severity": severity,
            "description": description,
            "machineId": machine_id,
            "telemetry": telemetry or {},
            "aiPredictions": ai_predictions or {},
            "resolvedAt": None
        }

        result = self._make_request("POST", endpoint, data=data)
        if result and "objectId" in result:
            self._last_event_time = now
            self.last_status = "Connected & Logging"
            self.last_error = ""
            return result
        return None

    def log_maintenance(self, action, notes="", machine_id="esp32_node_01"):
        """
        Create a MaintenanceLog record in Back4App.

        Args:
            action     : str — e.g., 'BATTERY_REPLACED', 'CALIBRATION',
                         'SENSOR_CLEANED', 'SYSTEM_RESTART'
            notes      : str — optional maintenance notes
            machine_id : str — machine identifier

        Returns:
            dict with 'objectId' and 'createdAt' on success, None on failure.
        """
        if not self.enabled:
            return None

        class_name = self.config.get("classes", {}).get("maintenance_log", "MaintenanceLog")
        endpoint = f"/classes/{class_name}"

        data = {
            "action": action,
            "notes": notes,
            "machineId": machine_id,
            "performedBy": "dashboard_user"
        }

        result = self._make_request("POST", endpoint, data=data)
        if result and "objectId" in result:
            self.last_status = "Connected & Logging"
            self.last_error = ""
            return result
        return None

    def get_recent_events(self, limit=20):
        """
        Query recent MachineEvent records from Back4App.

        Args:
            limit: maximum number of events to retrieve (default 20)

        Returns:
            List of event dicts, or empty list on failure.
        """
        if not self.enabled:
            return []

        class_name = self.config.get("classes", {}).get("machine_event", "MachineEvent")
        endpoint = f"/classes/{class_name}"

        params = {
            "limit": str(limit),
            "order": "-createdAt"  # newest first
        }

        result = self._make_request("GET", endpoint, params=params)
        if result and "results" in result:
            return result["results"]
        return []

    def get_maintenance_history(self, limit=10):
        """
        Query maintenance log records from Back4App.

        Args:
            limit: maximum number of records to retrieve (default 10)

        Returns:
            List of maintenance dicts, or empty list on failure.
        """
        if not self.enabled:
            return []

        class_name = self.config.get("classes", {}).get("maintenance_log", "MaintenanceLog")
        endpoint = f"/classes/{class_name}"

        params = {
            "limit": str(limit),
            "order": "-createdAt"
        }

        result = self._make_request("GET", endpoint, params=params)
        if result and "results" in result:
            return result["results"]
        return []

    def get_event_count(self, hours=24):
        """
        Get the count of events in the last N hours.

        Returns:
            int: number of events, or 0 on failure.
        """
        if not self.enabled:
            return 0

        class_name = self.config.get("classes", {}).get("machine_event", "MachineEvent")
        endpoint = f"/classes/{class_name}"

        # Parse Server date query: createdAt >= (now - hours)
        import datetime
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
        cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        where = json.dumps({
            "createdAt": {
                "$gte": {"__type": "Date", "iso": cutoff_iso}
            }
        })

        params = {
            "where": where,
            "limit": "0",
            "count": "1"
        }

        result = self._make_request("GET", endpoint, params=params)
        if result and "count" in result:
            return result["count"]
        return 0
