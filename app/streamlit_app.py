"""
ANPR Traffic Dashboard - Streamlit application for traffic monitoring.
Displays: Live vehicle counts, License plate logs, Traffic analytics, Historical reports
"""

import streamlit as st
import pandas as pd
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="Smart Traffic ANPR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database path
DB_PATH = "database/traffic.db"
CAPTURED_DIR = "output/captured"


def init_session_state():
    """Initialize Streamlit session state."""
    if 'db_path' not in st.session_state:
        st.session_state.db_path = DB_PATH
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5


# ============ Database Functions (embedded) ============

def get_db_connection(db_path):
    """Get database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_crossings_db(db_path, start_time=None, end_time=None, vehicle_type=None, license_plate=None, limit=1000):
    """Query crossing records."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM crossings WHERE 1=1"
    params = []

    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)
    if vehicle_type:
        query += " AND vehicle_type = ?"
        params.append(vehicle_type)
    if license_plate:
        query += " AND license_plate = ?"
        params.append(license_plate)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def get_stats_db(db_path) -> Dict[str, Any]:
    """Get statistics from database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Total
    cursor.execute("SELECT COUNT(*) FROM crossings")
    total = cursor.fetchone()[0]

    # Unique plates
    cursor.execute("""
        SELECT COUNT(DISTINCT license_plate) FROM crossings
        WHERE license_plate IS NOT NULL AND license_plate != ''
    """)
    unique = cursor.fetchone()[0]

    # By type
    cursor.execute("SELECT vehicle_type, COUNT(*) as cnt FROM crossings GROUP BY vehicle_type")
    by_type = {row[0]: row[1] for row in cursor.fetchall()}

    # By direction
    cursor.execute("SELECT direction, COUNT(*) as cnt FROM crossings WHERE direction IS NOT NULL GROUP BY direction")
    by_direction = {row[0]: row[1] for row in cursor.fetchall()}

    return {
        'total_crossings': total,
        'unique_plates': unique,
        'by_vehicle_type': by_type,
        'by_direction': by_direction
    }


def get_recent_plates_db(db_path, limit=10):
    """Get recent plates."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT vehicle_type, license_plate, timestamp, direction
        FROM crossings
        WHERE license_plate IS NOT NULL AND license_plate != ''
        ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    return [dict(row) for row in cursor.fetchall()]


def export_csv_db(db_path, output_path, start_time=None, end_time=None):
    """Export to CSV."""
    records = get_crossings_db(db_path, start_time, end_time, limit=100000)
    if not records:
        return 0

    with open(output_path, 'w', newline='') as f:
        if records:
            f.write(','.join(records[0].keys()) + '\n')
            for r in records:
                f.write(','.join(str(v) for v in r.values()) + '\n')
    return len(records)


# ============ Dashboard Functions ============

def get_vehicle_type_icon(vehicle_type: str) -> str:
    """Get emoji icon for vehicle type."""
    icons = {'car': '🚗', 'truck': '🚚', 'bus': '🚌', 'motorbike': '🏍️'}
    return icons.get(vehicle_type.lower(), '🚙')


def ensure_database(db_path):
    """Ensure database exists with schema."""
    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS crossings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_type TEXT NOT NULL,
                license_plate TEXT,
                timestamp TEXT NOT NULL,
                frame_idx INTEGER,
                direction TEXT,
                track_id INTEGER,
                confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()


def get_stats() -> Dict[str, Any]:
    """Get statistics from database."""
    try:
        db_path = st.session_state.get('db_path', DB_PATH)
    except:
        db_path = DB_PATH

    ensure_database(db_path)

    stats = get_stats_db(db_path)
    stats['recent_plates'] = get_recent_plates_db(db_path, 10)
    return stats


def get_crossings(start_time=None, end_time=None, vehicle_type=None, license_plate=None, limit=100):
    """Get crossing records."""
    try:
        db_path = st.session_state.get('db_path', DB_PATH)
    except:
        db_path = DB_PATH

    if not os.path.exists(db_path):
        return []

    return get_crossings_db(db_path, start_time, end_time, vehicle_type, license_plate, limit)


def main():
    """Main dashboard function."""
    init_session_state()

    st.title("🚗 Smart Traffic ANPR Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Settings")
    db_path = st.sidebar.text_input("Database Path", DB_PATH)
    st.session_state.db_path = db_path

    refresh = st.sidebar.slider("Refresh interval (seconds)", 1, 60, 5)
    st.session_state.refresh_interval = refresh
    if refresh > 0:
        st.sidebar.markdown(f"🔄 Auto-refresh every {refresh}s")

    st.markdown("---")

    # Load stats
    try:
        stats = get_stats()
    except Exception as e:
        st.error(f"Database error: {e}")
        stats = {'total_crossings': 0, 'unique_plates': 0, 'by_vehicle_type': {}, 'by_direction': {}, 'recent_plates': []}

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Crossings", stats.get('total_crossings', 0))
    with col2:
        st.metric("Unique Plates", stats.get('unique_plates', 0))
    with col3:
        cars = stats.get('by_vehicle_type', {}).get('car', 0)
        st.metric("Cars", cars)
    with col4:
        others = sum(stats.get('by_vehicle_type', {}).values()) - cars
        st.metric("Other Vehicles", others)

    st.markdown("---")

    # Live counts
    st.header("📊 Live Vehicle Counts")
    by_type = stats.get('by_vehicle_type', {})

    if by_type:
        cols = st.columns(len(by_type))
        for i, (vehicle_type, count) in enumerate(sorted(by_type.items(), key=lambda x: -x[1])):
            with cols[i % 4]:
                icon = get_vehicle_type_icon(vehicle_type)
                st.metric(f"{icon} {vehicle_type.title()}", count)
    else:
        st.info("No vehicles detected yet. Process a video to see counts.")

    # Charts
    if by_type:
        st.subheader("Vehicle Type Distribution")
        chart_data = pd.DataFrame({
            'Vehicle Type': [v.title() for v in by_type.keys()],
            'Count': list(by_type.values())
        }).set_index('Vehicle Type')
        st.bar_chart(chart_data['Count'])

    by_direction = stats.get('by_direction', {})
    if by_direction:
        st.subheader("Traffic Direction")
        dir_cols = st.columns(len(by_direction))
        for i, (direction, count) in enumerate(by_direction.items()):
            with dir_cols[i]:
                icon = "⬆️" if direction == "up" else "⬇️"
                st.metric(f"{icon} {direction.title()}", count)

    st.markdown("---")

    # Plate logs
    st.header("📋 License Plate Logs")
    recent_plates = stats.get('recent_plates', [])

    if recent_plates:
        df = pd.DataFrame(recent_plates)
        df['icon'] = df['vehicle_type'].apply(get_vehicle_type_icon)
        st.dataframe(df[['icon', 'vehicle_type', 'license_plate', 'timestamp', 'direction']])
    else:
        st.info("No license plates recorded yet.")

    st.markdown("---")

    # Analytics
    st.header("📈 Traffic Analytics")
    all_crossings = get_crossings(limit=1000)

    if all_crossings:
        df = pd.DataFrame(all_crossings)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        st.subheader("Hourly Traffic")
        df['hour'] = df['timestamp'].dt.hour
        hourly = df.groupby('hour').size()
        st.line_chart(hourly)

        st.subheader("Vehicle Type Timeline")
        for vtype in df['vehicle_type'].unique():
            vtype_data = df[df['vehicle_type'] == vtype]
            vtype_counts = vtype_data.groupby('hour').size()
            st.text(f"{get_vehicle_type_icon(vtype)} {vtype.title()}")
    else:
        st.info("No data for analytics.")

    st.markdown("---")

    # Historical reports
    st.header("📑 Historical Reports")
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))
    with date_col2:
        end_date = st.date_input("End Date", datetime.now().date())

    if st.button("📥 Export to CSV"):
        try:
            output_path = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            start_ts = datetime.combine(start_date, datetime.min.time()).isoformat()
            end_ts = datetime.combine(end_date, datetime.max.time()).isoformat()
            count = export_csv_db(st.session_state.db_path, output_path, start_ts, end_ts)
            st.success(f"Exported {count} records to {output_path}")
        except Exception as e:
            st.error(f"Export error: {e}")

    st.markdown("---")

    # Captured images
    st.header("📷 Captured Plates")
    if os.path.exists(CAPTURED_DIR):
        images = [f for f in os.listdir(CAPTURED_DIR) if f.endswith('.jpg')]
        if images:
            selected = st.selectbox("Select", images)
            if selected:
                st.image(os.path.join(CAPTURED_DIR, selected))
        else:
            st.info("No plates captured yet.")
    else:
        st.info("No captured plates directory found.")

    st.markdown("---")
    st.markdown("🚗 Smart Traffic ANPR System | Powered by YOLOv8 + EasyOCR")


if __name__ == "__main__":
    main()