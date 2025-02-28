import os
from abc import ABC, abstractmethod
import sqlite3
from tqdm import tqdm
import time

def get_analysis_database_repository() -> 'OptionDataRepository':
    """Erstellt und gibt eine Instanz von OptionDataRepository zurück."""
    filename = os.getenv('DB_ANALYSIS_FILENAME')
    db_path = os.getenv('DB_ANALYSIS_PATH')

    # Sicherstellen, dass der Ordner existiert
    if not os.path.exists(db_path):
        os.makedirs(db_path, exist_ok=True)

    full_db_path = os.path.join(db_path, filename)
    return OptionDataRepository(full_db_path)


class AnalysisDatabaseRepository(ABC):

    @abstractmethod
    def analysis_db_migrate(self):
        ...

    @abstractmethod
    def bulk_insert(self, data, table_name):
        ...


class OptionDataRepository(AnalysisDatabaseRepository):
    """Datenbankklasse für die Optionsdaten."""

    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path
        self.connection = self.get_analysis_database_connection()
        self.cursor = self.connection.cursor()

    def analysis_db_migrate(self):
        """Stellt sicher, dass alle Analyse-Tabellen existieren."""
        try:
            c = self.connection.cursor()

            table_schema = """
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                execution_price REAL,
                market_price_base REAL,
                remaining_time INTEGER,
                risk_free_interest REAL,
                trade_date TEXT,
                market_price_option REAL,
                implied_volatility REAL
            )
            """

            indices = os.getenv("INDICES_TO_ANALYZE", "").split(",")  # Indizes aus .env
            table_names = [f"sorted_{index.lower()}_data" for index in indices]

            for table in table_names:
                c.execute(table_schema.format(table_name=table))

            self.connection.commit()
            print(f"✅ Tabellen erstellt: {table_names}")
        except sqlite3.Error as e:
            print(f"❌ Fehler beim Erstellen der Tabellen: {e}")

    def bulk_insert(self, data, table_name):
        """Massiv optimierter Bulk-Insert mit Index-Checks und Batch-Processing."""
        try:
            start_time = time.time()
            print(f"🔄 Starte optimierten Bulk-Insert für {len(data)} Einträge in {table_name}...")

            # PRAGMA-Optimierungen
            self.cursor.execute("PRAGMA synchronous = OFF")
            self.cursor.execute("PRAGMA journal_mode = MEMORY")
            self.cursor.execute("PRAGMA temp_store = MEMORY")
            self.cursor.execute("PRAGMA cache_size = 100000")

            # Stelle sicher, dass Index existiert
            self.cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_ticker_trade_date ON {table_name} (ticker, trade_date);")

            with self.connection:
                self.cursor.execute("BEGIN TRANSACTION;")

                batch_size = 10000  # Kleinere Batches für effizienteres Arbeiten
                num_batches = (len(data) // batch_size) + (1 if len(data) % batch_size else 0)

                for batch_start in tqdm(range(0, len(data), batch_size), desc=f"Batch-Insert in {table_name}",
                                        unit="batch"):
                    batch = data[batch_start:batch_start + batch_size]

                    # Korrekte Anzahl an Platzhaltern für (ticker, trade_date)
                    placeholders = ",".join(["(?, ?)" for _ in batch])
                    query = f"SELECT ticker, trade_date FROM {table_name} WHERE (ticker, trade_date) IN ({placeholders})"

                    # Parameter als flache Liste übergeben (statt Liste von Tupeln)
                    self.cursor.execute(query, [val for row in batch for val in (row[0], row[5])])
                    existing_entries = set(self.cursor.fetchall())

                    # Batch-Update für existierende Einträge
                    update_query = f"""
                    UPDATE {table_name} SET 
                        execution_price = COALESCE(?, execution_price),
                        market_price_base = COALESCE(?, market_price_base),
                        remaining_time = COALESCE(?, remaining_time),
                        risk_free_interest = COALESCE(?, risk_free_interest),
                        market_price_option = COALESCE(?, market_price_option),
                        implied_volatility = COALESCE(?, implied_volatility)
                    WHERE ticker = ? AND trade_date = ?
                    """

                    self.cursor.executemany(update_query, [
                        (row[1], row[2], row[3], row[4], row[6], row[7], row[0], row[5])
                        for row in batch if (row[0], row[5]) in existing_entries
                    ])

                    # Batch-Insert für neue Einträge
                    insert_query = f"""
                    INSERT INTO {table_name} (
                        ticker, execution_price, market_price_base, remaining_time,
                        risk_free_interest, trade_date, market_price_option,
                        implied_volatility
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """

                    self.cursor.executemany(insert_query, [
                        row for row in batch if (row[0], row[5]) not in existing_entries
                    ])

                self.cursor.execute("COMMIT;")  # Transaktion abschließen

            total_time = time.time() - start_time
            print(f"✅ Bulk-Insert abgeschlossen für {table_name} in {total_time:.2f} Sekunden.")

        except sqlite3.Error as e:
            print(f"❌ Fehler beim Einfügen in {table_name}: {e}")

    def close(self):
        """Schließt die Datenbankverbindung."""
        self.connection.close()

    def get_analysis_database_connection(self):
        return sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
