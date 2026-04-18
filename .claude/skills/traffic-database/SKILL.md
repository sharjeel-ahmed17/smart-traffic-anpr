---
name: traffic-database
description: >
  Store and query vehicle crossing events (vehicle type, license plate, timestamp) in a SQLite or
  PostgreSQL database for the smart city traffic ANPR system. Use this skill whenever the user needs:
  store traffic data, save vehicle logs to database, create traffic DB schema, query crossing records,
  insert plate numbers into SQLite, export traffic logs to CSV, set up schema.sql for ANPR,
  traffic.db setup, vehicle event logging. This skill is used after license-plate-ocr produces
  a plate reading and needs to persist the result.
---

# Traffic Database Skill

Store structured vehicle crossing events in **SQLite** (default) or **PostgreSQL**.

---

## Database Schema

### SQLite Setup (Default)

```python
import sqlite3
from datetime import datetime

DB_PATH = "database/traffic.db"

def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS crossings (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_type  TEXT    NOT NULL,
            plate_number  TEXT    NOT NULL DEFAULT 'UNKNOWN',
            timestamp     TEXT    NOT NULL,
            direction     TEXT    DEFAULT 'down',
            confidence    REAL    DEFAULT 0.0,
            track_id      INTEGER DEFAULT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_timestamp    ON crossings(timestamp);
        CREATE INDEX IF NOT EXISTS idx_vehicle_type ON crossings(vehicle_type);
        CREATE INDEX IF NOT EXISTS idx_plate        ON crossings(plate_number);
    """)
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")
```

### schema.sql (save as `database/schema.sql`)

```sql
-- Smart Traffic ANPR Database Schema
CREATE TABLE IF NOT EXISTS crossings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_type  TEXT    NOT NULL,
    plate_number  TEXT    NOT NULL DEFAULT 'UNKNOWN',
    timestamp     TEXT    NOT NULL,
    direction     TEXT    DEFAULT 'down',
    confidence    REAL    DEFAULT 0.0,
    track_id      INTEGER DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_timestamp    ON crossings(timestamp);
CREATE INDEX IF NOT EXISTS idx_vehicle_type ON crossings(vehicle_type);
CREATE INDEX IF NOT EXISTS idx_plate        ON crossings(plate_number);
```

---

## Insert a Crossing Event

```python
def store_crossing(vehicle_type: str, plate_number: str,
                   direction: str = "down", confidence: float = 0.0,
                   track_id: int = None):
    """
    Insert a vehicle crossing event into the database.

    Args:
        vehicle_type: 'car', 'bike', 'truck', or 'bus'
        plate_number: OCR result e.g. 'ABC-123' or 'UNKNOWN'
        direction:    'up' or 'down'
        confidence:   detection confidence (0.0 - 1.0)
        track_id:     DeepSORT/ByteTrack ID (optional)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO crossings (vehicle_type, plate_number, timestamp, direction, confidence, track_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (vehicle_type, plate_number, timestamp, direction, confidence, track_id))
    conn.commit()
    conn.close()
```

---

## Query Helpers

```python
import pandas as pd

def get_all_crossings():
    """Return all records as a pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM crossings ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def get_count_by_type():
    """Return vehicle counts grouped by type."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT vehicle_type, COUNT(*) as count
        FROM crossings
        GROUP BY vehicle_type
        ORDER BY count DESC
    """, conn)
    conn.close()
    return df

def get_crossings_by_plate(plate_number: str):
    """Look up all events for a specific plate."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM crossings
        WHERE plate_number = ?
        ORDER BY timestamp DESC
    """, conn, params=(plate_number,))
    conn.close()
    return df

def get_crossings_in_range(start: str, end: str):
    """
    Get crossings between two timestamps.
    start/end format: 'YYYY-MM-DD HH:MM:SS'
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM crossings
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """, conn, params=(start, end))
    conn.close()
    return df

def get_hourly_traffic():
    """Aggregate traffic counts by hour."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            strftime('%Y-%m-%d %H:00', timestamp) AS hour,
            vehicle_type,
            COUNT(*) as count
        FROM crossings
        GROUP BY hour, vehicle_type
        ORDER BY hour ASC
    """, conn)
    conn.close()
    return df
```

---

## Export to CSV

```python
def export_to_csv(output_path: str = "output/logs.csv"):
    """Export all crossing records to a CSV file."""
    df = get_all_crossings()
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} records to {output_path}")
    return output_path
```

---

## PostgreSQL (Production Alternative)

For production deployments with multiple readers or high traffic volume:

```python
import psycopg2
import os

PG_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "database": os.getenv("PG_DB", "smart_traffic"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "")
}

def store_crossing_pg(vehicle_type, plate_number,
                      direction="down", confidence=0.0, track_id=None):
    timestamp = datetime.now()
    conn = psycopg2.connect(**PG_CONFIG)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO crossings (vehicle_type, plate_number, timestamp, direction, confidence, track_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (vehicle_type, plate_number, timestamp, direction, confidence, track_id))
    conn.commit()
    conn.close()
```

---

## Example Output

```
Vehicle Type | Plate Number | Timestamp
-------------|--------------|-------------------
Truck        | ABC-123      | 2024-01-15 10:01:00
Truck        | XYZ-456      | 2024-01-15 10:02:30
Car          | DEF-789      | 2024-01-15 10:05:12
Bus          | GHI-012      | 2024-01-15 10:07:45
```

---

## Notes

- Call `init_db()` once at application startup.
- Use `store_crossing()` inside the line-crossing callback (from **line-crossing-counter** skill).
- Use `get_all_crossings()` and `get_hourly_traffic()` to feed the **traffic-dashboard** skill.
- For full pipeline integration, see the **anpr-pipeline** skill.
