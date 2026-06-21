# Grafana Labs & ThingSpeak Setup Walkthrough

This guide provides step-by-step instructions to configure **ThingSpeak** as an IoT data collector and **Grafana Cloud** as your premium visualization dashboard.

---

## Part 1: Setting up ThingSpeak

### 1. Create a Channel
1. Go to [ThingSpeak](https://thingspeak.com/) and sign up or sign in.
2. Click **Channels** -> **My Channels** -> **New Channel**.
3. Fill in the channel details:
   - **Name**: `Machine Health Monitoring System`
   - **Description**: `IoT Cloud Telemetry & AI Preds`
4. Enable the following fields in order:
   - **Field 1**: `Temperature (C)`
   - **Field 2**: `Current (A)`
   - **Field 3**: `Vibration (RMS)`
   - **Field 4**: `RPM`
   - **Field 5**: `Battery (%)`
   - **Field 6**: `Fused Fault Probability (%)`
   - **Field 7**: `CNN Vibration Probability (%)`
   - **Field 8**: `RF Telemetry Probability (%)`
5. Click **Save Channel**.

### 2. Get API Keys
1. Navigate to the **API Keys** tab on your Channel page.
2. Note the:
   - **Channel ID** (displayed at the top of the page)
   - **Write API Key**
   - **Read API Key**
3. Open your local **Machine Health AI Dashboard** and enter these details in the **Cloud Sync** sidebar. Check **Enable Cloud Sync** and click **Save Cloud Config**.

---

## Part 2: Setting up Grafana Labs

We will use **Grafana Cloud** and the **Infinity** plugin (or JSON API) to read feeds directly from ThingSpeak.

### 1. Set up Grafana Cloud
1. Go to [Grafana Labs](https://grafana.com/) and sign up for a free Grafana Cloud account.
2. Once inside your Grafana workspace, click on the **Connections** icon in the sidebar, then select **Connect data**.

### 2. Install the Infinity Datasource
1. Search for `Infinity` (a popular, highly flexible JSON/CSV/GraphQL datasource plugin).
2. Install the plugin and click **Create a new instance of this data source**.
3. Name it `ThingSpeak Cloud`.
4. In the **Authentication** section, leave it as is, or set up default headers if needed (not required as we will pass the Read API Key in the URL query parameters).
5. Click **Save & test**.

### 3. Build a Dashboard
1. Go to **Dashboards** and click **New** -> **Dashboard** -> **Add a new panel**.
2. Select `ThingSpeak Cloud` (Infinity) as the Query Datasource.
3. In the query builder, configure:
   - **Type**: `JSON`
   - **Source**: `URL`
   - **Method**: `GET`
   - **URL**: `https://api.thingspeak.com/channels/<YOUR_CHANNEL_ID>/feeds.json?api_key=<YOUR_READ_API_KEY>&results=100`
4. Parse the JSON feeds array by setting:
   - **Parser**: `JSON`
   - **Root / Rows Path**: `feeds`
5. Map the fields under **Columns**:
   - Selector: `created_at` | Type: `DateTime` | Title: `Time`
   - Selector: `field1` | Type: `Number` | Title: `Temperature`
   - Selector: `field2` | Type: `Number` | Title: `Current`
   - Selector: `field3` | Type: `Number` | Title: `Vibration`
   - Selector: `field4` | Type: `Number` | Title: `RPM`
   - Selector: `field5` | Type: `Number` | Title: `Battery`
   - Selector: `field6` | Type: `Number` | Title: `Fused Fault Probability`
   - Selector: `field7` | Type: `Number` | Title: `CNN Probability`
   - Selector: `field8` | Type: `Number` | Title: `RF Probability`

### 4. Create Panels
Now configure your panels using Grafana's visualization options:
- **Telemetry Trends**: A **Time series** plot showing `Temperature`, `Current`, and `Vibration` over time.
- **System Health Status**: A **Stat** panel linked to `Fused Fault Probability`. If > 50%, color red (Critical), else color green (Normal).
- **Remaining Useful Life/Battery**: A **Gauge** panel for `Battery` value.
- **Model Confidences**: A **Bar gauge** displaying `CNN Probability` vs `RF Probability`.
