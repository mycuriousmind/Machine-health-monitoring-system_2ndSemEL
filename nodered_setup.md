# Node-RED, InfluxDB & Back4App — Setup Guide

This guide walks you through setting up all three integrations for the Machine Health Monitoring System.

---

## Prerequisites

- **Docker Desktop** installed on your machine ([Download](https://www.docker.com/products/docker-desktop/))
- **Back4App** free account ([Sign up](https://www.back4app.com/))
- Python 3.8+ with pip

---

## Part 1: Start Node-RED & InfluxDB (Docker)

1. Open a terminal in the project directory (`b:\2nd_Semester_EL\`).
2. Run Docker Compose:
   ```bash
   docker-compose up -d
   ```
3. Verify both containers are running:
   ```bash
   docker-compose ps
   ```
4. Access the UIs:
   - **Node-RED**: [http://localhost:1880](http://localhost:1880)
   - **InfluxDB**: [http://localhost:8086](http://localhost:8086) (User: `admin`, Password: `machinehealth123`)

### InfluxDB Token

The Docker Compose sets up a default admin token: `machine-health-influxdb-token`.
This is already pre-filled in `integration_config.json`. For production, generate a new token via the InfluxDB UI under **Load Data → API Tokens**.

---

## Part 2: Configure Node-RED

### 2.1 Install Required Palette Nodes

1. Open Node-RED at [http://localhost:1880](http://localhost:1880).
2. Click the **☰ Menu** (top-right) → **Manage palette**.
3. Go to the **Install** tab and search for:
   - `node-red-contrib-influxdb` — InfluxDB nodes for reading/writing
4. Click **Install** for each palette.

### 2.2 Import the Pre-Built Flow

1. Click **☰ Menu** → **Import**.
2. Select **Clipboard** tab.
3. Open `nodered_flows.json` from the project directory.
4. Copy the entire JSON contents and paste into the import dialog.
5. Click **Import** → **Deploy**.

### 2.3 Configure InfluxDB Connection

1. Double-click the **"Write to InfluxDB"** node in the flow.
2. Click the **pencil icon** next to the Server field to edit the InfluxDB config.
3. Verify or update:
   - **Version**: `2.0`
   - **URL**: `http://influxdb:8086` (if using Docker) or `http://localhost:8086` (if native)
   - **Token**: `machine-health-influxdb-token`
   - **Organisation**: `machine-health`
   - **Bucket**: `telemetry`
4. Click **Update** → **Done** → **Deploy**.

### 2.4 Configure Back4App Credentials in Node-RED

The flow uses Node-RED global context variables for Back4App credentials:

1. Open Node-RED at [http://localhost:1880](http://localhost:1880).
2. Go to **☰ Menu** → **Configuration nodes** → or use the **Sidebar** → **Context Data** → **Global**.
3. Alternatively, add an **Inject** node to set credentials at startup:
   - Create a new flow tab (or use the existing one).
   - Add an **Inject** node (set to inject once, after 0.1 seconds of start).
   - Add a **Change** node connected to it.
   - In the Change node, set:
     - `global.back4app_app_id` to your Back4App App ID
     - `global.back4app_rest_key` to your Back4App REST API Key
     - `global.thingspeak_write_key` to your ThingSpeak Write API Key
   - Deploy.

---

## Part 3: Set Up Back4App

### 3.1 Create an App

1. Go to [Back4App Dashboard](https://www.back4app.com/).
2. Click **Build new app** → **Backend as a Service**.
3. Name it: `MachineHealthMonitor`.
4. Click **Create**.

### 3.2 Get API Keys

1. In your app dashboard, go to **App Settings** → **Security & Keys**.
2. Note down:
   - **Application ID** (App ID)
   - **REST API Key**
3. Enter these values in `integration_config.json`:
   ```json
   "back4app": {
       "app_id": "YOUR_ACTUAL_APP_ID",
       "rest_api_key": "YOUR_ACTUAL_REST_API_KEY"
   }
   ```

### 3.3 Create Classes (Auto-Created on First Write)

The Parse Server will **automatically create** the `MachineEvent` and `MaintenanceLog` classes when the first record is written. The schemas will be:

**MachineEvent**:
| Column         | Type   | Description                        |
|----------------|--------|------------------------------------|
| eventType      | String | FAULT_DETECTED, TEMPERATURE_ALERT  |
| severity       | String | INFO, WARNING, CRITICAL            |
| description    | String | Human-readable event description   |
| machineId      | String | Machine identifier (esp32_node_01) |
| telemetry      | Object | Snapshot of sensor readings         |
| aiPredictions  | Object | AI model probabilities             |
| resolvedAt     | Date   | When the event was resolved        |

**MaintenanceLog**:
| Column       | Type   | Description                   |
|--------------|--------|-------------------------------|
| action       | String | BATTERY_REPLACED, CALIBRATION |
| notes        | String | Free-text maintenance notes   |
| machineId    | String | Machine identifier            |
| performedBy  | String | Who performed the maintenance |

---

## Part 4: Install Python Dependencies

```bash
pip install influxdb-client
```

The other modules (`back4app_manager.py`, `nodered_client.py`) use only Python stdlib (`urllib`).

---

## Part 5: Update Configuration

Edit `integration_config.json` with your actual credentials:

```json
{
    "nodered": {
        "enabled": true,
        "url": "http://localhost:1880",
        "telemetry_endpoint": "/api/telemetry",
        "alert_endpoint": "/api/alert"
    },
    "influxdb": {
        "enabled": true,
        "url": "http://localhost:8086",
        "org": "machine-health",
        "bucket": "telemetry",
        "token": "machine-health-influxdb-token"
    },
    "back4app": {
        "enabled": true,
        "server_url": "https://parseapi.back4app.com",
        "app_id": "YOUR_ACTUAL_APP_ID",
        "rest_api_key": "YOUR_ACTUAL_REST_API_KEY"
    }
}
```

---

## Part 6: Verify Everything Works

### Quick Test

Run the Streamlit dashboard in **Simulation mode**:
```bash
streamlit run dashboard.py
```

Then check:
1. **Node-RED Debug panel** — should show incoming telemetry messages
2. **InfluxDB UI** → Data Explorer → query `machine_telemetry` measurement
3. **Back4App Dashboard** → Browse `MachineEvent` class (events appear when faults are simulated)

### Connection Test Script

```bash
python -c "from influxdb_manager import InfluxDBManager; m = InfluxDBManager(); print('InfluxDB:', 'OK' if m.test_connection() else m.last_error)"
python -c "from back4app_manager import Back4AppManager; m = Back4AppManager(); print('Back4App:', 'OK' if m.test_connection() else m.last_error)"
python -c "from nodered_client import NodeREDClient; c = NodeREDClient(); print('Node-RED:', 'OK' if c.test_connection() else c.last_error)"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Node-RED not reachable on :1880 | Check `docker-compose ps`, ensure container is running |
| InfluxDB auth errors | Verify token in `integration_config.json` matches Docker setup |
| Back4App 401 errors | Double-check App ID and REST API Key in config |
| `influxdb-client` import error | Run `pip install influxdb-client` |
| Node-RED InfluxDB node error | Install `node-red-contrib-influxdb` palette in Node-RED UI |
| Docker Compose version warning | Update Docker Desktop or use `version: "3.8"` in docker-compose.yml |
