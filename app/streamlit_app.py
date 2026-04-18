"""
ANPR Traffic Dashboard - Streamlit application for traffic monitoring.
Displays: Live vehicle counts, License plate logs, Traffic analytics, Historical reports
"""

import streamlit as st
import pandas as pd
import os
import cv2
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


def load_database():
    """Load database module."""
    try:
        db_path = st.session_state.get('db_path', DB_PATH)
    except:
        db_path = DB_PATH
    import sys
    sys.path.insert(0, "..")
    from traffic_database import TrafficDatabase
    return TrafficDatabase(db_path)


def get_stats() -> Dict[str, Any]:
    """Get statistics from database."""
    # Use default path if not in streamlit
    try:
        db_path = st.session_state.get('db_path', DB_PATH)
    except:
        db_path = DB_PATH
    from traffic_database import TrafficDatabase
    db = TrafficDatabase(db_path)
    stats = db.get_stats()
    stats['recent_plates'] = db.get_recent_plates(10)
    return stats


def get_vehicle_type_icon(vehicle_type: str) -> str:
    """Get emoji icon for vehicle type."""
    icons = {
        'car': '🚗',
        'truck': '🚚',
        'bus': '🚌',
        'motorbike': '🏍️'
    }
    return icons.get(vehicle_type.lower(), '🚙')


def main():
    """Main dashboard function."""
    init_session_state()

    st.title("🚗 Smart Traffic ANPR Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Settings")

    # Database path
    db_path = st.sidebar.text_input("Database Path", DB_PATH)
    st.session_state.db_path = db_path

    # Refresh interval
    refresh = st.sidebar.slider("Auto-refresh (seconds)", 1, 60, 5)
    st.session_state.refresh_interval = refresh

    # Auto-refresh
    if refresh > 0:
        st.sidebar.markdown(f"🔄 Auto-refresh every {refresh}s")
        st.autorefresh(refresh * 1000)

    # ============== MAIN CONTENT ==============

    # Statistics section
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

    # ============== LIVE COUNTS ==============

    st.header("📊 Live Vehicle Counts")

    # Vehicle type breakdown
    by_type = stats.get('by_vehicle_type', {})

    if by_type:
        cols = st.columns(len(by_type))
        for i, (vehicle_type, count) in enumerate(sorted(by_type.items(), key=lambda x: -x[1])):
            with cols[i % 4]:
                icon = get_vehicle_type_icon(vehicle_type)
                st.metric(f"{icon} {vehicle_type.title()}", count)
    else:
        st.info("No vehicles detected yet. Process a video to see counts.")

    # ============== VEHICLE TYPE CHART ==============

    if by_type:
        st.subheader("Vehicle Type Distribution")

        # Bar chart
        chart_data = pd.DataFrame({
            'Vehicle Type': [v.title() for v in by_type.keys()],
            'Count': list(by_type.values())
        }).set_index('Vehicle Type')

        st.bar_chart(chart_data['Count'])

    # Direction breakdown
    by_direction = stats.get('by_direction', {})
    if by_direction:
        st.subheader("Traffic Direction")
        dir_cols = st.columns(len(by_direction))
        for i, (direction, count) in enumerate(by_direction.items()):
            with dir_cols[i]:
                icon = "⬆️" if direction == "up" else "⬇️"
                st.metric(f"{icon} {direction.title()}", count)

    st.markdown("---")

    # ============== LICENSE PLATE LOGS ==============

    st.header("📋 License Plate Logs")

    # Recent plates table
    recent_plates = stats.get('recent_plates', [])

    if recent_plates:
        # Create DataFrame
        df = pd.DataFrame(recent_plates)
        df['icon'] = df['vehicle_type'].apply(get_vehicle_type_icon)

        # Display
        st.dataframe(
            df[['icon', 'vehicle_type', 'license_plate', 'timestamp', 'direction']],
            use_container_width=True,
            hide_index=True
        )

        # Plate details
        with st.expander("View All Plate Details"):
            st.table(df)
    else:
        st.info("No license plates recorded. Process video with ANPR pipeline.")

    # ============== SEARCH PLATES ==============

    st.subheader("🔍 Search Plates")

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        plate_search = st.text_input("Search by plate number", "")
    with search_col2:
        vehicle_filter = st.selectbox("Filter by vehicle type",
                                    ["All"] + list(by_type.keys()))

    if plate_search or vehicle_filter != "All":
        try:
            db = load_database()
            filter_type = None if vehicle_filter == "All" else vehicle_filter
            results = db.get_crossings(license_plate=plate_search if plate_search else None,
                                     vehicle_type=filter_type,
                                     limit=50)
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else:
                st.info("No matching records found.")
        except Exception as e:
            st.error(f"Search error: {e}")

    st.markdown("---")

    # ============== TRAFFIC ANALYTICS ==============

    st.header("📈 Traffic Analytics")

    # Time-based analysis
    try:
        db = load_database()
        all_crossings = db.get_crossings(limit=1000)
    except:
        all_crossings = []

    if all_crossings:
        # Convert to DataFrame
        df = pd.DataFrame(all_crossings)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Time grouping
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date

        # Hourly distribution
        st.subheader("Hourly Traffic")

        hourly = df.groupby('hour').size()
        st.line_chart(hourly)

        # Daily distribution (if multiple days)
        if df['date'].nunique() > 1:
            st.subheader("Daily Traffic")

            daily = df.groupby('date').size()
            st.bar_chart(daily)

        # Vehicle type over time
        st.subheader("Vehicle Type Timeline")

        for vtype in df['vehicle_type'].unique():
            vtype_data = df[df['vehicle_type'] == vtype]
            vtype_counts = vtype_data.groupby('hour').size()
            st.text(f"{get_vehicle_type_icon(vtype)} {vtype.title()}")
    else:
        st.info("No data for analytics. Process more videos.")

    st.markdown("---")

    # ============== HISTORICAL REPORTS ==============

    st.header("📑 Historical Reports")

    # Date range selector
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))
    with date_col2:
        end_date = st.date_input("End Date", datetime.now().date())

    # Get data for date range
    try:
        db = load_database()
        start_ts = datetime.combine(start_date, datetime.min.time()).isoformat()
        end_ts = datetime.combine(end_date, datetime.max.time()).isoformat()

        records = db.get_crossings(start_time=start_ts, end_time=end_ts, limit=10000)
    except:
        records = []

    # Summary for date range
    st.subheader(f"Report: {start_date} to {end_date}")

    if records:
        df_range = pd.DataFrame(records)
        df_range['timestamp'] = pd.to_datetime(df_range['timestamp'])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Crossings", len(records))

        with col2:
            st.metric("Unique Plates", df_range['license_plate'].nunique())

        with col3:
            st.metric("Peak Hour", df_range['timestamp'].dt.hour.mode()[0] if not df_range.empty else "-")

        # Breakdown
        breakdown = df_range['vehicle_type'].value_counts()
        st.bar_chart(breakdown)

        # Export button
        if st.button("📥 Export to CSV"):
            try:
                db = load_database()
                output_path = f"database/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                count = db.export_csv(output_path, start_ts, end_ts)
                st.success(f"Exported {count} records to {output_path}")
            except Exception as e:
                st.error(f"Export error: {e}")

    else:
        st.info("No records for selected date range.")

    st.markdown("---")

    # ============== CAPTURED IMAGES ==============

    st.header("📷 Captured Plates")

    if os.path.exists(CAPTURED_DIR):
        images = [f for f in os.listdir(CAPTURED_DIR) if f.endswith('.jpg')]

        if images:
            # Select image
            selected_image = st.selectbox("Select captured plate", images)

            if selected_image:
                image_path = os.path.join(CAPTURED_DIR, selected_image)
                st.image(image_path, caption=selected_image, use_container_width=True)

                # Get metadata
                parts = selected_image.replace('.jpg', '').split('_')
                if len(parts) >= 3:
                    track_id = parts[1]
                    frame_idx = parts[2]
                    st.text(f"Track ID: {track_id}, Frame: {frame_idx}")
        else:
            st.info("No plates captured yet.")
    else:
        st.info("Captured plates directory not found.")

    st.markdown("---")

    # Footer
    st.footer("🚗 Smart Traffic ANPR System | Powered by YOLOv8 + EasyOCR")


if __name__ == "__main__":
    # Add autorefresh support
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5000, key="refresh")
    except ImportError:
        pass

    main()