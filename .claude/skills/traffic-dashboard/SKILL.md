---
name: traffic-dashboard
description: >
  Build a Streamlit dashboard for the smart city traffic ANPR system showing live vehicle counts,
  license plate logs, traffic analytics graphs, and historical reports. Use this skill whenever the
  user needs: Streamlit traffic dashboard, visualize vehicle counts, display plate logs, traffic
  analytics UI, real-time traffic monitoring app, build dashboard for ANPR system, FastAPI traffic
  service, traffic graphs and reports. Reads from the database created by the traffic-database skill.
---

# Traffic Dashboard Skill

Build a **Streamlit** dashboard (or **FastAPI** service) to display live vehicle counts, plate logs,
and traffic analytics from the ANPR database.

---

## Dependencies

```bash
pip install streamlit pandas plotly sqlite3
# For FastAPI service:
pip install fastapi uvicorn
```

---

## Streamlit App (`app/streamlit_app.py`)

### Full Dashboard Code

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta

DB_PATH = "database/traffic.db"

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Smart Traffic Monitor",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart City Traffic Monitoring Dashboard")
st.markdown("Real-time vehicle detection, counting, and license plate recognition.")

# ─── Auto-refresh ─────────────────────────────────────────
refresh_rate = st.sidebar.slider("Auto-refresh (seconds)", 5, 60, 10)
st.sidebar.write(f"Next refresh in {refresh_rate}s")

# ─── Load Data ────────────────────────────────────────────
@st.cache_data(ttl=refresh_rate)
def load_crossings():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM crossings ORDER BY timestamp DESC", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_crossings()

# ─── Live Counters ────────────────────────────────────────
st.subheader("📊 Live Vehicle Counts")

col1, col2, col3, col4 = st.columns(4)
counts = df["vehicle_type"].value_counts()

col1.metric("🚗 Cars",   counts.get("car",   0))
col2.metric("🏍️ Bikes",  counts.get("bike",  0))
col3.metric("🚛 Trucks", counts.get("truck", 0))
col4.metric("🚌 Buses",  counts.get("bus",   0))

st.divider()

# ─── Charts ───────────────────────────────────────────────
st.subheader("📈 Traffic Analytics")

tab1, tab2, tab3 = st.tabs(["Hourly Traffic", "Vehicle Breakdown", "Timeline"])

with tab1:
    # Hourly traffic volume
    df_hourly = df.copy()
    df_hourly["hour"] = df_hourly["timestamp"].dt.floor("H")
    hourly = df_hourly.groupby(["hour", "vehicle_type"]).size().reset_index(name="count")
    if not hourly.empty:
        fig = px.bar(hourly, x="hour", y="count", color="vehicle_type",
                     title="Vehicles Per Hour",
                     labels={"hour": "Time", "count": "Count", "vehicle_type": "Type"},
                     color_discrete_map={
                         "car": "#00CC44", "bike": "#FF8C00",
                         "truck": "#0044FF", "bus": "#CC00CC"
                     })
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet.")

with tab2:
    # Pie chart by vehicle type
    if not df.empty:
        type_counts = df["vehicle_type"].value_counts().reset_index()
        type_counts.columns = ["Vehicle Type", "Count"]
        fig2 = px.pie(type_counts, values="Count", names="Vehicle Type",
                      title="Vehicle Type Distribution",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # Rolling 5-min traffic volume
    if not df.empty:
        df_time = df.set_index("timestamp").resample("5min").size().reset_index(name="count")
        fig3 = px.line(df_time, x="timestamp", y="count",
                       title="Traffic Volume (5-min intervals)")
        st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ─── License Plate Log ─────────────────────────────────────
st.subheader("🔍 License Plate Log")

# Filter controls
col_a, col_b, col_c = st.columns(3)
with col_a:
    search_plate = st.text_input("Search plate number", "")
with col_b:
    type_filter = st.multiselect("Vehicle type", ["car", "bike", "truck", "bus"],
                                 default=["car", "bike", "truck", "bus"])
with col_c:
    hours_back = st.slider("Show last N hours", 1, 72, 24)

# Apply filters
cutoff = datetime.now() - timedelta(hours=hours_back)
filtered = df[
    (df["timestamp"] >= cutoff) &
    (df["vehicle_type"].isin(type_filter))
]
if search_plate:
    filtered = filtered[filtered["plate_number"].str.contains(
        search_plate.upper(), na=False)]

# Display table
st.dataframe(
    filtered[["timestamp", "vehicle_type", "plate_number", "direction"]]
    .rename(columns={
        "timestamp": "Time",
        "vehicle_type": "Type",
        "plate_number": "Plate",
        "direction": "Dir"
    })
    .reset_index(drop=True),
    use_container_width=True
)

st.caption(f"Showing {len(filtered)} records")

st.divider()

# ─── Historical Report ─────────────────────────────────────
st.subheader("📋 Historical Report")

if st.button("📥 Export CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download logs.csv",
        data=csv,
        file_name=f"traffic_logs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
```

---

## Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`

---

## FastAPI REST Service (Alternative / Companion)

For exposing traffic data as a JSON API:

```python
# app/api.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import sqlite3, pandas as pd
from datetime import datetime

app = FastAPI(title="Smart Traffic API")
DB_PATH = "database/traffic.db"

@app.get("/crossings")
def get_crossings(limit: int = Query(100, le=1000)):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM crossings ORDER BY timestamp DESC LIMIT {limit}", conn)
    conn.close()
    return df.to_dict(orient="records")

@app.get("/counts")
def get_counts():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT vehicle_type, COUNT(*) as count
        FROM crossings GROUP BY vehicle_type
    """, conn)
    conn.close()
    return df.to_dict(orient="records")

@app.get("/plate/{plate_number}")
def lookup_plate(plate_number: str):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM crossings WHERE plate_number = ? ORDER BY timestamp DESC",
        conn, params=(plate_number.upper(),))
    conn.close()
    return df.to_dict(orient="records")
```

```bash
uvicorn app.api:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

---

## Dashboard Features Summary

| Feature              | Implementation                    |
| -------------------- | --------------------------------- |
| Live vehicle counts  | Metric cards with auto-refresh    |
| Hourly traffic chart | Plotly stacked bar chart          |
| Type distribution    | Plotly pie chart                  |
| Timeline view        | Plotly line chart (5-min buckets) |
| Plate log table      | Filterable dataframe              |
| CSV export           | Download button                   |
| REST API             | FastAPI endpoints                 |

---

## Notes

- Set `ttl=refresh_rate` in `@st.cache_data` so the dashboard auto-refreshes from the DB.
- For live video overlay, run the ANPR pipeline in a separate process; dashboard reads the same DB.
- See the **anpr-pipeline** skill for integrating all components end-to-end.
