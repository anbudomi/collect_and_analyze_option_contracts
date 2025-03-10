import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

# ------------------------------
# PreparedDataRepository (bestehende Version)
# ------------------------------

def get_prepared_database_repository() -> 'PreparedDataRepository':
    """Erstellt und gibt eine Instanz von PreparedDataRepository zurück. Löscht die bestehende DB, falls vorhanden."""
    prepared_db_filename = os.getenv('PREPARED_DB_FILENAME')
    prepared_db_path = os.getenv('PREPARED_DB_PATH')

    if not os.path.exists(prepared_db_path):
        os.makedirs(prepared_db_path, exist_ok=True)

    full_db_path = os.path.join(prepared_db_path, prepared_db_filename)

    # Falls die Datenbank bereits existiert, löschen
    if os.path.exists(full_db_path):
        os.remove(full_db_path)

    return PreparedDataRepository(full_db_path)


class PreparedDatabaseRepository(ABC):
    @abstractmethod
    def prepared_data_migrate(self, index):
        ...


class PreparedDataRepository(PreparedDatabaseRepository):
    """Datenbankklasse für vorbereitete Optionsdaten."""
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = self.get_connection()
        self.cursor = self.connection.cursor()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def prepared_data_migrate(self, index):
        """Erstellt eine Tabelle für den angegebenen Index, falls sie nicht existiert."""
        table_name = f"prepared_{index.lower()}_data"
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                date TEXT,
                option_type TEXT,
                execution_price REAL,
                market_base_price REAL,
                remaining_days INTEGER,
                remaining_time REAL,
                risk_free_rate REAL,
                market_price_option REAL,
                implied_vola_percent REAL,
                implied_vola_dec REAL,
                dividend_yield REAL,
                BSM REAL,
                absolute_error REAL,
                relative_error REAL,
                moneyness REAL,
                UNIQUE(ticker, date)
            )
        """)
        self.connection.commit()

    def bulk_insert(self, data, index):
        """Führt einen Bulk-Insert für vorbereitete Daten aus."""
        table_name = f"prepared_{index.lower()}_data"
        insert_query = f"""
            INSERT INTO {table_name} (
                ticker, execution_price, market_base_price, remaining_days,
                risk_free_rate, date, market_price_option, implied_vola_percent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            self.cursor.executemany(insert_query, data)
            self.connection.commit()
            print(f"✅ {len(data)} Datensätze erfolgreich in {table_name} eingefügt.")
        except sqlite3.Error as e:
            print(f"⚠ Fehler beim Einfügen der Daten in {table_name}: {e}")
            self.connection.rollback()

    def close(self):
        """Schließt die Datenbankverbindung."""
        self.connection.close()


# ------------------------------
# PrefilteredDataRepository (neu)
# ------------------------------

def get_prefiltered_database_repository() -> 'PrefilteredDataRepository':
    """Erstellt und gibt eine Instanz von PrefilteredDataRepository zurück. Löscht die bestehende DB, falls vorhanden."""
    prefiltered_db_filename = os.getenv('PREFILTERED_DB_FILENAME')
    prefiltered_db_path = os.getenv('PREFILTERED_DB_PATH')

    if not os.path.exists(prefiltered_db_path):
        os.makedirs(prefiltered_db_path, exist_ok=True)

    full_db_path = os.path.join(prefiltered_db_path, prefiltered_db_filename)

    if os.path.exists(full_db_path):
        os.remove(full_db_path)

    return PrefilteredDataRepository(full_db_path)


class PrefilteredDataRepository:
    """Datenbankklasse für vorgefilterte Optionsdaten."""
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = self.get_connection()
        self.cursor = self.connection.cursor()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def prefiltered_data_migrate(self, ticker):
        """Erstellt eine Tabelle für den angegebenen Ticker, falls sie nicht existiert."""
        table_name = f"prefiltered_{ticker.lower()}_data"
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                underlying_ticker TEXT,
                date TEXT,
                contract_type TEXT,
                expiration_date TEXT,
                remaining_days INTEGER,
                remaining_time REAL,
                execution_price REAL,
                market_price_base REAL,
                market_price_option REAL,
                implied_vola_percent REAL,
                implied_vola_dec REAL,
                risk_free_rate REAL,
                UNIQUE(ticker, date)
            )
        """)
        self.connection.commit()

    def bulk_insert_prefiltered(self, data, ticker):
        """Führt einen Bulk-Insert für vorgefilterte Daten in die Tabelle für den angegebenen Ticker aus."""
        table_name = f"prefiltered_{ticker.lower()}_data"
        insert_query = f"""
            INSERT INTO {table_name} (
                ticker, underlying_ticker, date, contract_type, expiration_date,
                remaining_days, remaining_time, execution_price, market_price_base,
                market_price_option, implied_vola_percent, implied_vola_dec, risk_free_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            self.cursor.executemany(insert_query, data)
            self.connection.commit()
            print(f"✅ {len(data)} Datensätze erfolgreich in {table_name} eingefügt.")
        except sqlite3.Error as e:
            print(f"⚠ Fehler beim Einfügen der Daten in {table_name}: {e}")
            self.connection.rollback()

    def close(self):
        """Schließt die Datenbankverbindung."""
        self.connection.close()
