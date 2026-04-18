"""
Traffic Database - SQLite storage for vehicle crossing events.
Stores: Vehicle type, License plate number, Timestamp
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


# Default database path
DEFAULT_DB_PATH = "database/traffic.db"


class TrafficDatabase:
    """
    SQLite database for storing ANPR traffic events.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize traffic database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create crossings table
            cursor.execute("""
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

            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON crossings(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vehicle_type
                ON crossings(vehicle_type)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_plate
                ON crossings(license_plate)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def add_crossing(self,
                   vehicle_type: str,
                   license_plate: Optional[str] = None,
                   timestamp: Optional[str] = None,
                   frame_idx: Optional[int] = None,
                   direction: Optional[str] = None,
                   track_id: Optional[int] = None,
                   confidence: Optional[float] = None) -> int:
        """
        Add a crossing event to the database.

        Args:
            vehicle_type: Type of vehicle (car, truck, bus, motorbike)
            license_plate: Detected license plate number
            timestamp: Timestamp of crossing (default: now)
            frame_idx: Frame index when crossing occurred
            direction: Crossing direction (up, down)
            track_id: Unique track ID
            confidence: Detection confidence

        Returns:
            Row ID of inserted record
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO crossings
                (vehicle_type, license_plate, timestamp, frame_idx, direction, track_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (vehicle_type, license_plate, timestamp, frame_idx, direction, track_id, confidence))

            conn.commit()
            return cursor.lastrowid

    def add_crossing_from_event(self, event: Dict[str, Any]) -> int:
        """
        Add crossing from event dictionary.

        Args:
            event: Event dict with required keys

        Returns:
            Row ID of inserted record
        """
        return self.add_crossing(
            vehicle_type=event.get("class_name"),
            license_plate=event.get("license_plate"),
            timestamp=event.get("timestamp"),
            frame_idx=event.get("frame_idx"),
            direction=event.get("direction"),
            track_id=event.get("track_id"),
            confidence=event.get("confidence")
        )

    def get_crossings(self,
                      start_time: Optional[str] = None,
                      end_time: Optional[str] = None,
                      vehicle_type: Optional[str] = None,
                      license_plate: Optional[str] = None,
                      limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Query crossing records.

        Args:
            start_time: Filter start timestamp (ISO format)
            end_time: Filter end timestamp
            vehicle_type: Filter by vehicle type
            license_plate: Filter by plate number
            limit: Max records to return

        Returns:
            List of crossing records
        """
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

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            return [dict(row) for row in cursor.fetchall()]

    def get_counts(self, group_by: str = "vehicle_type") -> Dict[str, int]:
        """
        Get crossing counts.

        Args:
            group_by: Group by 'vehicle_type', 'direction', or 'license_plate'

        Returns:
            Dictionary of counts
        """
        if group_by == "vehicle_type":
            column = "vehicle_type"
        elif group_by == "direction":
            column = "direction"
        elif group_by == "license_plate":
            column = "license_plate"
        else:
            raise ValueError(f"Invalid group_by: {group_by}")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT {column}, COUNT(*) as count
                FROM crossings
                WHERE {column} IS NOT NULL
                GROUP BY {column}
                ORDER BY count DESC
            """)

            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_recent_plates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent license plate sightings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT vehicle_type, license_plate, timestamp, direction
                FROM crossings
                WHERE license_plate IS NOT NULL AND license_plate != ''
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_vehicle_history(self, license_plate: str) -> List[Dict[str, Any]]:
        """Get crossing history for a specific plate."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM crossings
                WHERE license_plate = ?
                ORDER BY timestamp DESC
            """, (license_plate,))

            return [dict(row) for row in cursor.fetchall()]

    def export_csv(self, output_path: str,
                start_time: Optional[str] = None,
                end_time: Optional[str] = None) -> int:
        """
        Export records to CSV.

        Args:
            output_path: Path to output CSV file
            start_time: Filter start timestamp
            end_time: Filter end timestamp

        Returns:
            Number of records exported
        """
        records = self.get_crossings(start_time, end_time, limit=100000)

        if not records:
            return 0

        # Write CSV
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if records:
                fieldnames = records[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)

        return len(records)

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total crossings
            cursor.execute("SELECT COUNT(*) FROM crossings")
            total = cursor.fetchone()[0]

            # Unique plates
            cursor.execute("""
                SELECT COUNT(DISTINCT license_plate)
                FROM crossings
                WHERE license_plate IS NOT NULL AND license_plate != ''
            """)
            unique_plates = cursor.fetchone()[0]

            # By vehicle type
            cursor.execute("""
                SELECT vehicle_type, COUNT(*) as count
                FROM crossings
                GROUP BY vehicle_type
                ORDER BY count DESC
            """)
            by_type = {row[0]: row[1] for row in cursor.fetchall()}

            # By direction
            cursor.execute("""
                SELECT direction, COUNT(*) as count
                FROM crossings
                WHERE direction IS NOT NULL
                GROUP BY direction
            """)
            by_direction = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_crossings": total,
                "unique_plates": unique_plates,
                "by_vehicle_type": by_type,
                "by_direction": by_direction
            }

    def clear(self):
        """Clear all records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM crossings")
            conn.commit()

    def delete_old(self, days: int = 30) -> int:
        """
        Delete records older than N days.

        Args:
            days: Number of days to keep

        Returns:
            Number of records deleted
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM crossings WHERE timestamp < ?", (cutoff_str,))
            conn.commit()
            return cursor.rowcount


def init_db(db_path: str = DEFAULT_DB_PATH) -> TrafficDatabase:
    """Initialize a new database."""
    return TrafficDatabase(db_path)


def add_event(vehicle_type: str, license_plate: str = None,
              db_path: str = DEFAULT_DB_PATH) -> int:
    """Convenience function to add an event."""
    db = TrafficDatabase(db_path)
    return db.add_crossing(vehicle_type, license_plate)


def query_events(**kwargs) -> List[Dict[str, Any]]:
    """Convenience function to query events."""
    db = TrafficDatabase(kwargs.get("db_path", DEFAULT_DB_PATH))
    return db.get_crossings(
        start_time=kwargs.get("start_time"),
        end_time=kwargs.get("end_time"),
        vehicle_type=kwargs.get("vehicle_type"),
        license_plate=kwargs.get("license_plate"),
        limit=kwargs.get("limit", 1000)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Traffic Database CLI")
    parser.add_argument("--db", default=DEFAULT_DB_PATH,
                      help="Database path")
    parser.add_argument("--add", nargs=3, metavar=("TYPE", "PLATE", "TIMESTAMP"),
                      help="Add crossing event")
    parser.add_argument("--query", action="store_true",
                      help="Query recent crossings")
    parser.add_argument("--stats", action="store_true",
                      help="Show statistics")
    parser.add_argument("--export", metavar="FILE",
                      help="Export to CSV")
    parser.add_argument("--clear", action="store_true",
                      help="Clear database")
    args = parser.parse_args()

    db = TrafficDatabase(args.db)

    if args.add:
        vehicle_type, license_plate, timestamp = args.add
        db.add_crossing(vehicle_type, license_plate, timestamp)
        print(f"Added: {vehicle_type} - {license_plate}")

    elif args.stats:
        stats = db.get_stats()
        print("\nTraffic Database Statistics:")
        print(f"  Total crossings: {stats['total_crossings']}")
        print(f"  Unique plates: {stats['unique_plates']}")
        print("  By vehicle type:")
        for vt, count in stats['by_vehicle_type'].items():
            print(f"    {vt}: {count}")
        print("  By direction:")
        for direction, count in stats['by_direction'].items():
            print(f"    {direction}: {count}")

    elif args.export:
        count = db.export_csv(args.export)
        print(f"Exported {count} records to {args.export}")

    elif args.clear:
        db.clear()
        print("Database cleared")

    else:
        # Default: show recent
        crossings = db.get_crossings(limit=20)
        print(f"\nRecent crossings ({len(crossings)} records):")
        for c in crossings:
            print(f"  {c['timestamp']} | {c['vehicle_type']} | {c['license_plate'] or '-'} | {c['direction'] or '-'}")