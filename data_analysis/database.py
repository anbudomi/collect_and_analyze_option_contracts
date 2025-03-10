import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta


def get_prepared_database_repository() -> 'PreparedDataRepository':
    """Erstellt und gibt eine Instanz von PreparedDataRepository zurück. Löscht die bestehende DB, falls vorhanden."""
    filename = os.getenv('PREPARED_DB_FILENAME')
    db_path = os.getenv('PREPARED_DB_PATH')

    if not os.path.exists(db_path):
        os.makedirs(db_path, exist_ok=True)

    full_db_path = os.path.join(db_path, filename)

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
        self.connection = self.get_prepared_database_connection()
        self.cursor = self.connection.cursor()

    def get_prepared_database_connection(self):
        connection = sqlite3.connect(self.db_path)
        return connection

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
                UNIQUE(ticker, date)  -- Stellt sicher, dass eine Kombination aus date und ticker nur einmal vorkommt
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