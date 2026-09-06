"""
Database Manager (IBVAP)
Manages SQLite storage for tactical watchlists, geofence zones,
and immutable event audit trails with automatic schema migration.
"""

import sqlite3
import datetime
from thefuzz import fuzz


class DBManager:
    def __init__(self, db_path="surveillance.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()
        self._migrate_tables()
        self._seed_default_tactical_data()

    def _create_tables(self):
        c = self.conn.cursor()
        
        # 1. Vehicle Watchlist (ANPR)
        c.execute('''CREATE TABLE IF NOT EXISTS vehicle_watchlist (
            plate TEXT PRIMARY KEY,
            vehicle_type TEXT,
            owner_name TEXT,
            reason TEXT,
            threat_level TEXT
        )''')

        # 2. Person of Interest Watchlist (FRS)
        c.execute('''CREATE TABLE IF NOT EXISTS person_watchlist (
            person_id TEXT PRIMARY KEY,
            name TEXT,
            alias TEXT,
            threat_level TEXT,
            reason TEXT,
            photo_path TEXT
        )''')

        # 3. Comprehensive Incident & Surveillance Event Log
        c.execute('''CREATE TABLE IF NOT EXISTS event_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            camera_id TEXT,
            sector TEXT,
            event_type TEXT,
            subject_type TEXT,
            threat_level TEXT,
            description TEXT,
            plate TEXT,
            face_name TEXT,
            tx_hash TEXT,
            image_hash TEXT,
            evidence_image TEXT,
            status TEXT DEFAULT 'ACTIVE'
        )''')

        # 4. Virtual Fence & Zone Configurations
        c.execute('''CREATE TABLE IF NOT EXISTS zone_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT,
            zone_name TEXT,
            zone_type TEXT,
            coordinates TEXT,
            active INTEGER DEFAULT 1
        )''')

        self.conn.commit()

    def _migrate_tables(self):
        """Ensures all columns exist in event_log and watchlists even if upgrading an old DB."""
        c = self.conn.cursor()

        # Migrate event_log
        c.execute("PRAGMA table_info(event_log)")
        existing_cols = {row[1] for row in c.fetchall()}

        needed_cols = {
            "camera_id": "TEXT",
            "sector": "TEXT",
            "event_type": "TEXT",
            "subject_type": "TEXT",
            "threat_level": "TEXT",
            "description": "TEXT",
            "plate": "TEXT",
            "face_name": "TEXT",
            "tx_hash": "TEXT",
            "image_hash": "TEXT",
            "evidence_image": "TEXT",
            "status": "TEXT DEFAULT 'ACTIVE'"
        }

        for col_name, col_type in needed_cols.items():
            if col_name not in existing_cols:
                try:
                    c.execute(f"ALTER TABLE event_log ADD COLUMN {col_name} {col_type}")
                except Exception as e:
                    print(f"DB Migration Notice: {e}")

        # Migrate vehicle_watchlist
        c.execute("PRAGMA table_info(vehicle_watchlist)")
        vw_cols = {row[1] for row in c.fetchall()}
        for col, col_type in [("vehicle_type", "TEXT"), ("owner_name", "TEXT"), ("reason", "TEXT"), ("threat_level", "TEXT")]:
            if col not in vw_cols:
                try:
                    c.execute(f"ALTER TABLE vehicle_watchlist ADD COLUMN {col} {col_type}")
                except Exception:
                    pass

        # If legacy 'watchlist' exists, import any existing records into 'vehicle_watchlist'
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='watchlist'")
        if c.fetchone():
            try:
                c.execute('''INSERT OR IGNORE INTO vehicle_watchlist (plate, reason, threat_level)
                             SELECT plate, reason, threat_level FROM watchlist''')
            except Exception:
                pass

        self.conn.commit()

    def _seed_default_tactical_data(self):
        """Populates database with realistic operational border security profiles."""
        c = self.conn.cursor()

        # Seed Vehicles
        vehicles = [
            ("DL01AB1234", "Patrol Gypsy", "BSF 42nd Battalion", "Authorized Border Patrol Vehicle", "SAFE"),
            ("MH02CD5678", "Logistics Truck", "SSB Quartermaster", "Official Supply Carrier", "SAFE"),
            ("KA03EF9012", "Ambulance", "Border Medical Corps", "Emergency Response Unit", "SAFE"),
            ("HR26DK8901", "Heavy Truck", "Unregistered Carrier", "Known Cross-Border Arms Smuggling Vehicle", "CRITICAL"),
            ("PB10XY2345", "Black SUV", "Unknown Operator", "Narcotics Transport Suspect", "HIGH"),
            ("RJ14ZZ9876", "Pickup Truck", "Illegal Syndicate", "Unchecked Border Highway Intruder", "HIGH"),
            ("JK02AZ9999", "Off-Road 4x4", "Blacklisted Fleet", "Perimeter Reconnaissance Vehicle", "CRITICAL")
        ]
        for p, vt, ow, rs, th in vehicles:
            c.execute('''INSERT OR IGNORE INTO vehicle_watchlist (plate, vehicle_type, owner_name, reason, threat_level)
                         VALUES (?, ?, ?, ?, ?)''', (p, vt, ow, rs, th))

        # Seed Persons of Interest
        persons = [
            ("WANTED_01", "Tariq Rehman", "Shadow-01", "CRITICAL", "Cross-Border Infiltrator / Red Notice", "assets/faces/WANTED_01.jpg"),
            ("WANTED_02", "Vikram Singhania", "Smuggler-V", "HIGH", "Arms & Contraband Syndicate Leader", "assets/faces/WANTED_02.jpg"),
            ("AUTH_01", "Inspector R. Verma", "Eagle-Leader", "SAFE", "BSF Sector 7 Commander", "assets/faces/AUTH_01.jpg"),
            ("AUTH_02", "Constable M. Singh", "Sentinel-04", "SAFE", "Border Outpost Guard", "assets/faces/AUTH_02.jpg")
        ]
        for pid, nm, al, th, rs, pp in persons:
            c.execute('''INSERT OR IGNORE INTO person_watchlist (person_id, name, alias, threat_level, reason, photo_path)
                         VALUES (?, ?, ?, ?, ?, ?)''', (pid, nm, al, th, rs, pp))

        self.conn.commit()

    def check_plate(self, plate_text):
        """Checks plate against watchlist with exact and fuzzy matching."""
        clean_plate = plate_text.replace(" ", "").upper()
        c = self.conn.cursor()

        # Exact match
        c.execute("SELECT plate, vehicle_type, owner_name, reason, threat_level FROM vehicle_watchlist WHERE plate = ?", (clean_plate,))
        res = c.fetchone()
        if res:
            return {
                "status": "ALERT" if res[4] in ["HIGH", "CRITICAL"] else "CLEAR",
                "plate": res[0],
                "vehicle_type": res[1] or "Vehicle",
                "owner": res[2] or "Unknown",
                "reason": res[3] or "Flagged in Watchlist",
                "threat": res[4] or "HIGH",
                "match_type": "EXACT"
            }

        # Fuzzy match fallback for minor OCR variations
        c.execute("SELECT plate, vehicle_type, owner_name, reason, threat_level FROM vehicle_watchlist")
        for row in c.fetchall():
            ratio = fuzz.ratio(clean_plate, row[0])
            if ratio >= 82:
                return {
                    "status": "ALERT" if row[4] in ["HIGH", "CRITICAL"] else "CLEAR",
                    "plate": row[0],
                    "vehicle_type": row[1] or "Vehicle",
                    "owner": row[2] or "Unknown",
                    "reason": f"{row[3]} (Fuzzy Match: {ratio}%)",
                    "threat": row[4] or "HIGH",
                    "match_type": "FUZZY"
                }

        return {
            "status": "UNKNOWN",
            "plate": clean_plate,
            "vehicle_type": "Civilian Vehicle",
            "owner": "Unregistered",
            "reason": "Unregistered Vehicle at Border Checkpoint",
            "threat": "LOW",
            "match_type": "NONE"
        }

    def check_person(self, person_id_or_name):
        """Checks person against watchlist."""
        c = self.conn.cursor()
        c.execute("SELECT person_id, name, alias, threat_level, reason FROM person_watchlist WHERE person_id = ? OR name = ?", 
                  (person_id_or_name, person_id_or_name))
        res = c.fetchone()
        if res:
            return {
                "matched": True,
                "person_id": res[0],
                "name": res[1],
                "alias": res[2],
                "threat": res[3],
                "reason": res[4]
            }
        return {"matched": False, "threat": "LOW", "reason": "Unregistered Individual"}

    def log_event(self, camera_id, sector, event_type, subject_type, threat_level, description,
                  plate=None, face_name=None, tx_hash=None, image_hash=None, evidence_image=None):
        """Logs surveillance incident to audit ledger."""
        c = self.conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            c.execute('''INSERT INTO event_log (timestamp, camera_id, sector, event_type, subject_type,
                                                threat_level, description, plate, face_name,
                                                tx_hash, image_hash, evidence_image)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (timestamp, camera_id, sector, event_type, subject_type, threat_level, description,
                       plate, face_name, tx_hash, image_hash, evidence_image))
            self.conn.commit()
            return c.lastrowid
        except Exception as e:
            print(f"Error logging event: {e}")
            return 0

    def get_recent_events(self, limit=50, threat_filter=None):
        """Fetches recent events with optional threat level filter."""
        c = self.conn.cursor()
        try:
            if threat_filter and threat_filter != "ALL":
                c.execute("SELECT * FROM event_log WHERE threat_level = ? ORDER BY id DESC LIMIT ?", (threat_filter, limit))
            else:
                c.execute("SELECT * FROM event_log ORDER BY id DESC LIMIT ?", (limit,))
            
            cols = [d[0] for d in c.description]
            rows = c.fetchall()
            return [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            print(f"Error fetching recent events: {e}")
            return []

    def get_analytics_summary(self):
        """Computes executive situational awareness metrics safely."""
        c = self.conn.cursor()
        summary = {
            "total_events": 0,
            "critical_threats": 0,
            "high_threats": 0,
            "intrusions": 0,
            "watchlist_vehicles": 0
        }

        try:
            c.execute("SELECT COUNT(*) FROM event_log")
            row = c.fetchone()
            summary["total_events"] = row[0] if row else 0
        except Exception:
            pass

        try:
            c.execute("SELECT COUNT(*) FROM event_log WHERE threat_level = 'CRITICAL'")
            row = c.fetchone()
            summary["critical_threats"] = row[0] if row else 0
        except Exception:
            pass

        try:
            c.execute("SELECT COUNT(*) FROM event_log WHERE threat_level = 'HIGH'")
            row = c.fetchone()
            summary["high_threats"] = row[0] if row else 0
        except Exception:
            pass

        try:
            c.execute("SELECT COUNT(*) FROM event_log WHERE event_type LIKE '%INTRUSION%' OR event_type LIKE '%BREACH%'")
            row = c.fetchone()
            summary["intrusions"] = row[0] if row else 0
        except Exception:
            pass

        try:
            c.execute("SELECT COUNT(*) FROM vehicle_watchlist")
            row = c.fetchone()
            summary["watchlist_vehicles"] = row[0] if row else 0
        except Exception:
            pass

        return summary

    def add_vehicle_watchlist(self, plate, vehicle_type, owner_name, reason, threat_level):
        c = self.conn.cursor()
        c.execute('''INSERT OR REPLACE INTO vehicle_watchlist (plate, vehicle_type, owner_name, reason, threat_level)
                     VALUES (?, ?, ?, ?, ?)''', (plate.replace(" ", "").upper(), vehicle_type, owner_name, reason, threat_level))
        self.conn.commit()

    def add_person_watchlist(self, person_id, name, alias, threat_level, reason, photo_path=""):
        c = self.conn.cursor()
        c.execute('''INSERT OR REPLACE INTO person_watchlist (person_id, name, alias, threat_level, reason, photo_path)
                     VALUES (?, ?, ?, ?, ?, ?)''', (person_id, name, alias, threat_level, reason, photo_path))
        self.conn.commit()