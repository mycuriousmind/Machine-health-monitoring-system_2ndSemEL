import os
import json
import time
import urllib.request
import urllib.parse

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_config.json")

class CloudManager:
    def __init__(self):
        self.config = self.load_config()
        self.last_upload_time = 0.0
        self.last_status = "Not Connected"
        self.last_error = ""

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "enabled": False,
            "write_api_key": "",
            "channel_id": "",
            "read_api_key": "",
            "update_interval": 15
        }

    def save_config(self, enabled, write_api_key, channel_id, read_api_key, update_interval=15):
        self.config = {
            "enabled": enabled,
            "write_api_key": write_api_key.strip(),
            "channel_id": channel_id.strip(),
            "read_api_key": read_api_key.strip(),
            "update_interval": int(update_interval)
        }
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"[CloudManager] Error saving config: {e}")

    def upload_telemetry(self, temp, current, vibration, rpm, battery, fused_prob, cnn_prob, rf_prob):
        if not self.config.get("enabled", False):
            self.last_status = "Disabled"
            return False

        write_key = self.config.get("write_api_key", "")
        if not write_key:
            self.last_status = "Error: Write API Key missing"
            return False

        now = time.time()
        interval = self.config.get("update_interval", 15)
        
        # Enforce rate limiting
        if now - self.last_upload_time < interval:
            # Skip update, but don't flag as error
            return False

        # Build query parameters according to fields:
        # field1: Temp, field2: Current, field3: Vibration, field4: RPM, 
        # field5: Battery, field6: Fused Health Prob, field7: CNN Prob, field8: RF Prob
        params = {
            "api_key": write_key,
            "field1": f"{temp:.2f}",
            "field2": f"{current:.2f}",
            "field3": f"{vibration:.4f}",
            "field4": f"{rpm:.1f}",
            "field5": f"{battery:.1f}",
            "field6": f"{fused_prob * 100:.1f}",
            "field7": f"{cnn_prob * 100:.1f}",
            "field8": f"{rf_prob * 100:.1f}"
        }
        
        url = "https://api.thingspeak.com/update"
        data = urllib.parse.urlencode(params).encode('utf-8')
        
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=5) as response:
                resp_data = response.read().decode('utf-8')
                # ThingSpeak returns entry ID (an integer > 0) on success, or "0" on failure/rate limit
                if resp_data.strip() == "0":
                    self.last_status = "Rate Limited / Update Rejected"
                    self.last_error = "ThingSpeak rejected the request. Please check API Key or wait."
                    return False
                else:
                    self.last_upload_time = now
                    self.last_status = "Connected & Syncing"
                    self.last_error = ""
                    return True
        except Exception as e:
            self.last_status = "Network Error"
            self.last_error = str(e)
            return False
