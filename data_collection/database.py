import sqlite3
import os
from abc import ABC, abstractmethod
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
import time
from tqdm import tqdm


class DatabaseType(Enum):
    SQLITE = 'sqlite'


def get_collection_database_repository(type: DatabaseType) -> dict:
    if type == DatabaseType.SQLITE:
        # Alle Underlying-Ticker aus der .env lesen und bereinigen (Großbuchstaben)
        underlying_tickers = os.getenv('POLYGON_UNDERLYING_TICKER', 'UNKNOWN').replace(" ", "").upper().split(",")

        # Dictionary, das für jeden Ticker eine eigene Datenbank-Repository speichert
        repositories = {}

        for ticker in underlying_tickers:
            # Dynamisch den DB-Filename setzen (KEIN contract_type im Namen)
            db_filename = f"data_collection/data/rawdata_{ticker.lower()}_db.sqlite"
            repositories[ticker] = SqliteDatabaseRepository(db_filename)

        return repositories
    else:
        raise ValueError("Invalid database type.")




class CollectionDatabaseRepository(ABC):
    @abstractmethod
    def collection_db_migrate(self):
        ...

    @abstractmethod
    def insert_contract(self, contract):
        ...

    @abstractmethod
    def insert_contracts_bulk(self, contracts):
        ...

    @abstractmethod
    def insert_aggregate(self, aggregate):
        ...

    @abstractmethod
    def insert_contract_aggregates_bulk(self, aggregates):
        ...

    @abstractmethod
    def count_contracts(self):
        ...

    @abstractmethod
    def get_tickers(self, batch_size, offset):
        ...

    @abstractmethod
    def aggregate_exists(self, ticker, from_date, to_date):
        ...

    @abstractmethod
    def insert_data_of_index(self, data, ticker):
        ...

    @abstractmethod
    def insert_data_interest_rates(self, series_data, series_id):
        ...

    @abstractmethod
    def insert_data_implied_volatility(self, ticker, data):
        ...

class SqliteDatabaseRepository(CollectionDatabaseRepository):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.connection = self.get_collection_database_connection()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(sqlite3.OperationalError),
        reraise=True
    )

    def get_collection_database_connection(self):
        """
        Returns a single, reusable connection to the SQLite database.
        """
        conn = sqlite3.connect(self.filename, isolation_level=None, check_same_thread=False)
        c = conn.cursor()
        c.execute("PRAGMA synchronous = OFF")
        c.execute("PRAGMA journal_mode = MEMORY")
        c.execute("PRAGMA cache_size = -64000")
        return conn

    def collection_db_migrate(self):
        """
        Creates the tables if they don't already exist. Uses the single connection.
        """
        try:
            c = self.connection.cursor()

            # 'contracts' table
            c.execute("""
                CREATE TABLE IF NOT EXISTS contracts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    underlying_ticker TEXT,
                    cfi TEXT,
                    contract_type TEXT,
                    exercise_style TEXT,
                    expiration_date TEXT,
                    primary_exchange TEXT,
                    shares_per_contract INTEGER,
                    strike_price REAL,
                    date TEXT,
                    UNIQUE(ticker, date)
                )
            """)

            # 'contract_aggregates' table
            c.execute("""
                CREATE TABLE IF NOT EXISTS contract_aggregates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_ticker TEXT,
                    c REAL,
                    h REAL,
                    l REAL,
                    n INTEGER,
                    o REAL,
                    t INTEGER,
                    v INTEGER,
                    vw REAL,
                    date TEXT,
                    UNIQUE(contract_ticker, date)
                )
            """)

            # 'SP500_daily_close' table
            c.execute("""
                CREATE TABLE IF NOT EXISTS index_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    date TEXT,
                    close REAL,
                    high REAL,
                    low REAL,
                    open REAL,
                    volume INTEGER,
                    UNIQUE(ticker, date)
                )
            """)

            # 'treasury_bill' table
            c.execute("""
                CREATE TABLE IF NOT EXISTS treasury_bill (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    interest_rate REAL,
                    series_id TEXT,
                    UNIQUE(date, series_id)
                )
            """)

            # 'implicit_volatility' table
            c.execute("""
                CREATE TABLE IF NOT EXISTS implied_volatility (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    date TEXT,
                    close REAL,
                    high REAL,
                    low REAL,
                    open REAL,
                    volume INTEGER,
                    UNIQUE(ticker, date)
                )
            """)

            print("Database migration completed successfully.")
        except sqlite3.Error as e:
            print(f"An error occurred during migration: {e}")

    def is_day_sampled(self, date_str):
        """Prüft, ob ein bestimmtes Datum bereits in der contracts-Tabelle existiert."""
        c = self.connection.cursor()
        query = "SELECT COUNT(*) FROM contracts WHERE date = ?"
        c.execute(query, (date_str,))
        return c.fetchone()[0] > 0

    def mark_day_sampled(self, date_str):
        """Markiert einen Tag als verarbeitet, um doppelte Anfragen zu verhindern."""
        c = self.connection.cursor()
        query = "INSERT OR IGNORE INTO sampled_days (date) VALUES (?)"
        c.execute(query, (date_str,))
        self.connection.commit()

    def insert_contract(self, contract):
        """
        Inserts a contract, skipping insertion if a duplicate (ticker, date) already exists.
        Reuses self.connection.
        """
        c = self.connection.cursor()
        c.execute("""
            SELECT 1 FROM contracts WHERE ticker = ? AND date = ?
        """, (contract.get("ticker"), contract.get("date")))

        if c.fetchone():
            print(f"Duplicate detected: Ticker {contract.get('ticker')} on {contract.get('date')}. Skipping insertion.")
            return

        c.execute("""
            INSERT INTO contracts (
                ticker, underlying_ticker, cfi, contract_type,
                exercise_style, expiration_date, primary_exchange,
                shares_per_contract, strike_price, date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            contract.get("ticker"),
            contract.get("underlying_ticker"),
            contract.get("cfi"),
            contract.get("contract_type"),
            contract.get("exercise_style"),
            contract.get("expiration_date"),
            contract.get("primary_exchange"),
            contract.get("shares_per_contract"),
            contract.get("strike_price"),
            contract.get("date")
        ))
        # Commit not strictly necessary with isolation_level=None, but can be done if desired:
        # self.connection.commit()

    def insert_contracts_bulk(self, contracts_list):
        """
        Optimierter Bulk-Insert für Optionsdaten in SQLite.
        - Nutzt große Batches (10.000)
        - Verwendet explizite Transaktionen (`BEGIN TRANSACTION` & `COMMIT`)
        - Konvertiert alle Daten in kompatible Typen (keine None-Werte)
        - Nutzt `PRAGMA`-Optimierungen für schnelles Schreiben
        """
        if not contracts_list:
            print("⚠️  Keine Contracts zum Einfügen.")
            return

        start_time = time.time()
        print(f"🔄 Starte Bulk-Insert für {len(contracts_list)} Contracts...")

        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)

        c = self.connection.cursor()

        # **🚀 PRAGMA für schnellere Insert-Performance**
        c.execute("PRAGMA synchronous = OFF")
        c.execute("PRAGMA journal_mode = MEMORY")
        c.execute("PRAGMA temp_store = MEMORY")
        c.execute("PRAGMA cache_size = 100000")
        c.execute("PRAGMA busy_timeout = 5000")  # Falls DB gesperrt ist, warte bis zu 5 Sekunden

        sql = """
            INSERT OR IGNORE INTO contracts (
                ticker, underlying_ticker, cfi, contract_type,
                exercise_style, expiration_date, primary_exchange,
                shares_per_contract, strike_price, date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            batch_size = 100000
            num_batches = (len(contracts_list) // batch_size) + (1 if len(contracts_list) % batch_size else 0)

            for batch_start in tqdm(range(0, len(contracts_list), batch_size), desc="Batch-Insert in contracts",
                                    unit="batch"):
                batch = contracts_list[batch_start: batch_start + batch_size]

                # **Konvertiere `None`-Werte & Datentypen für SQLite**
                data_tuples = [
                    (
                        str(cdict.get("ticker", "")),
                        str(cdict.get("underlying_ticker", "")),
                        str(cdict.get("cfi", "")),
                        str(cdict.get("contract_type", "")),
                        str(cdict.get("exercise_style", "")),
                        str(cdict.get("expiration_date", "")),
                        str(cdict.get("primary_exchange", "")),
                        int(cdict.get("shares_per_contract", 0)),
                        float(cdict.get("strike_price", 0.0)),
                        str(cdict.get("date", ""))
                    )
                    for cdict in batch
                ]

                c.executemany(sql, data_tuples)

            total_time = time.time() - start_time
            print(f"✅ Bulk-Insert abgeschlossen in {total_time:.2f} Sekunden für {len(contracts_list)} Einträge.")

        except sqlite3.Error as e:
            print(f"❌ Fehler beim Einfügen in contracts: {e}")
            self.connection.rollback()  # **Rollback bei Fehlern**

    def insert_aggregate(self, aggregate):
        """
        Inserts a contract aggregate, skipping insertion if a duplicate (contract_ticker, date) already exists.
        Reuses self.connection.
        """
        c = self.connection.cursor()
        c.execute("""
            SELECT 1 FROM contract_aggregates WHERE contract_ticker = ? AND date = ?
        """, (aggregate["contract_ticker"], aggregate["date"]))

        if c.fetchone():
            print(f"Duplicate detected: Ticker {aggregate['contract_ticker']} on {aggregate['date']}. Skipping insertion.")
            return

        c.execute("""
            INSERT INTO contract_aggregates (
                contract_ticker, c, h, l, n, o, t, v, vw, date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            aggregate["contract_ticker"],
            aggregate["c"],
            aggregate["h"],
            aggregate["l"],
            aggregate["n"],
            aggregate["o"],
            aggregate["t"],
            aggregate["v"],
            aggregate["vw"],
            aggregate["date"]
        ))
        # self.connection.commit()

    def insert_contract_aggregates_bulk(self, aggregates_list):
        """
        Optimierter Bulk-Insert für Contract-Aggregate-Daten in SQLite.
        - Nutzt große Batches (10.000)
        - Verwendet explizite Transaktionen (`BEGIN TRANSACTION` & `COMMIT`)
        - Konvertiert alle Daten in kompatible Typen (keine None-Werte)
        - Nutzt `PRAGMA`-Optimierungen für schnelles Schreiben
        """
        if not aggregates_list:
            print("⚠️  Keine Aggregates zum Einfügen.")
            return

        start_time = time.time()
        print(f"🔄 Starte Bulk-Insert für {len(aggregates_list)} Contract-Aggregates...")

        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)

        c = self.connection.cursor()

        # **🚀 PRAGMA für schnellere Insert-Performance**
        c.execute("PRAGMA synchronous = OFF")
        c.execute("PRAGMA journal_mode = MEMORY")
        c.execute("PRAGMA temp_store = MEMORY")
        c.execute("PRAGMA cache_size = 100000")
        c.execute("PRAGMA busy_timeout = 5000")  # Falls DB gesperrt ist, warte bis zu 5 Sekunden

        sql = """
            INSERT OR IGNORE INTO contract_aggregates (
                contract_ticker, c, h, l, n, o, t, v, vw, date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            batch_size = 100000  # Setze die Batch-Größe auf 100.000 für maximale Effizienz
            num_batches = (len(aggregates_list) // batch_size) + (1 if len(aggregates_list) % batch_size else 0)

            for batch_start in tqdm(range(0, len(aggregates_list), batch_size),
                                    desc="Batch-Insert in contract_aggregates",
                                    unit="batch"):
                batch = aggregates_list[batch_start: batch_start + batch_size]

                # **Konvertiere `None`-Werte & Datentypen für SQLite**
                data_tuples = [
                    (
                        str(adict.get("contract_ticker", "")),  # contract_ticker (TEXT)
                        float(adict.get("c", 0.0)),  # c (REAL)
                        float(adict.get("h", 0.0)),  # h (REAL)
                        float(adict.get("l", 0.0)),  # l (REAL)
                        int(adict.get("n", 0)),  # n (INTEGER)
                        float(adict.get("o", 0.0)),  # o (REAL)
                        int(adict.get("t", 0)),  # t (INTEGER)
                        int(adict.get("v", 0)),  # v (INTEGER)
                        float(adict.get("vw", 0.0)),  # vw (REAL)
                        str(adict.get("date", ""))  # date (TEXT)
                    )
                    for adict in batch
                ]

                c.executemany(sql, data_tuples)

            total_time = time.time() - start_time
            print(f"✅ Bulk-Insert abgeschlossen in {total_time:.2f} Sekunden für {len(aggregates_list)} Einträge.")

        except sqlite3.Error as e:
            print(f"❌ Fehler beim Einfügen in contract_aggregates: {e}")
            self.connection.rollback()  # **Rollback bei Fehlern**

    def aggregate_exists(self, ticker, from_date, to_date):
        """
        Checks if there's at least one entry for 'ticker' in 'contract_aggregates'
        within the given date range.
        """
        c = self.connection.cursor()
        c.execute("""
            SELECT 1 FROM contract_aggregates
            WHERE contract_ticker = ?
              AND date >= ?
              AND date <= ?
            LIMIT 1
        """, (ticker, from_date, to_date))
        return c.fetchone() is not None

    def drop_all_tables(self):
        """
        Drops all relevant tables.
        """
        c = self.connection.cursor()
        c.execute("DROP TABLE IF EXISTS contracts")
        c.execute("DROP TABLE IF EXISTS contract_aggregates")
        print("All tables have been dropped.")

    def insert_data_of_index(self, data, ticker):
        # Falls die Spalten ein MultiIndex sind, flache sie ab, indem du Level 0 (Attributnamen) verwendest.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Wenn das Datum im Index steckt, in eine Spalte umwandeln.
        if 'Date' not in data.columns:
            data = data.reset_index()
            # Falls der ursprüngliche Index keinen Namen hatte, heißt die neue Spalte "index".
            if 'Date' not in data.columns:
                data.rename(columns={"index": "Date"}, inplace=True)

        # Debug-Ausgabe: Zeige die Spaltennamen
        print("Spalten nach reset_index:", data.columns)

        c = self.connection.cursor()

        for _, row in data.iterrows():
            # Datum in einen String umwandeln (falls es ein Timestamp ist)
            if hasattr(row['Date'], 'strftime'):
                date_str = row['Date'].strftime('%Y-%m-%d')
            else:
                date_str = str(row['Date'])

            # Zugriff auf die einzelnen Spaltenwerte als skalare Werte
            close_val = row["Close"]
            high_val = row["High"]
            low_val = row["Low"]
            open_val = row["Open"]
            volume_val = row["Volume"]

            values = (
                ticker,  # ticker (String)
                date_str,  # Datum als String
                close_val,  # Schlusskurs
                high_val,  # Tageshoch
                low_val,  # Tagestief
                open_val,  # Eröffnungskurs
                volume_val  # Handelsvolumen
            )

            print("Einfügen:", values)

            # Beispielhafte SQL-Abfrage – passe die Tabelle und Platzhalter an deine Datenbank an
            sql = ("INSERT INTO index_data "
                   "(ticker, date, close, high, low, open, volume) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?)")
            try:
                c.execute(sql, values)
                self.connection.commit()
                print("Daten erfolgreich eingefügt.")
            except Exception as e:
                print("Fehler beim Einfügen der Daten:", e)
                self.connection.rollback()

        self.connection.commit()

    def insert_data_interest_rates(self, series_data, series_id):

        c  = self.connection.cursor()
        # Iteriere über die abgerufenen Daten
        for date, interest_rate in series_data.items():
            # Stelle sicher, dass das Datum als String im Format YYYY-MM-DD vorliegt
            if hasattr(date, 'strftime'):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)

            # Einfügen in die Tabelle; bei doppelten (date, series_id)-Kombinationen wird der Eintrag ignoriert
            c.execute("""
                    INSERT OR IGNORE INTO treasury_bill (date, interest_rate, series_id)
                    VALUES (?, ?, ?)
                """, (date_str, interest_rate, series_id))

        self.connection.commit()

    def insert_data_implied_volatility(self, ticker, data):

        print("Ticker type:", type(ticker), ticker)

        # Falls die Spalten ein MultiIndex sind, flache sie ab, indem du Level 0 (Attributnamen) verwendest.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Wenn das Datum im Index steckt, in eine Spalte umwandeln.
        if 'Date' not in data.columns:
            data = data.reset_index()
            # Falls der ursprüngliche Index keinen Namen hatte, heißt die neue Spalte "index".
            if 'Date' not in data.columns:
                data.rename(columns={"index": "Date"}, inplace=True)

        # Debug-Ausgabe: Zeige die Spaltennamen
        print("Spalten nach reset_index:", data.columns)

        c = self.connection.cursor()

        for _, row in data.iterrows():
            # Datum in einen String umwandeln (falls es ein Timestamp ist)
            if hasattr(row['Date'], 'strftime'):
                date_str = row['Date'].strftime('%Y-%m-%d')
            else:
                date_str = str(row['Date'])

            # Zugriff auf die einzelnen Spaltenwerte als skalare Werte
            close_val = row["Close"]
            high_val = row["High"]
            low_val = row["Low"]
            open_val = row["Open"]
            volume_val = row["Volume"]

            values = (
                ticker,  # ticker (String)
                date_str,  # Datum als String
                close_val,  # Schlusskurs
                high_val,  # Tageshoch
                low_val,  # Tagestief
                open_val,  # Eröffnungskurs
                volume_val  # Handelsvolumen
            )

            print("Einfügen:", values)

            # Beispielhafte SQL-Abfrage – passe die Tabelle und Platzhalter an deine Datenbank an
            sql = ("INSERT INTO implied_volatility "
                   "(ticker, date, close, high, low, open, volume) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?)")
            try:
                c.execute(sql, values)
                self.connection.commit()
                print("Daten erfolgreich eingefügt.")
            except Exception as e:
                print("Fehler beim Einfügen der Daten:", e)
                self.connection.rollback()

        self.connection.commit()



    def fetch_all(self, query, params=()):
        """ Führt eine SQL-Select-Abfrage aus und gibt die Ergebnisse zurück. """
        c = self.connection.cursor()
        try:
            c.execute(query, params)
            return c.fetchall()
        except sqlite3.Error as e:
            print(f"❌ Fehler beim Abfragen der Datenbank: {e}")
            return []

    def count_contracts(self):
        """
        Returns the number of entries in the 'contracts' table, reusing the single connection.
        """
        c = self.connection.cursor()
        c.execute("SELECT COUNT(*) FROM contracts")
        count = c.fetchone()[0]
        return count

    def get_tickers(self, batch_size, offset):
        """
        Returns a list of (ticker, date, expiration_date) from 'contracts' in ascending date order.
        Uses LIMIT + OFFSET for pagination.
        """
        c = self.connection.cursor()
        c.execute("""
            SELECT DISTINCT ticker, date, expiration_date
            FROM contracts
            ORDER BY date ASC
            LIMIT ? OFFSET ?
        """, (batch_size, offset))
        tickers = c.fetchall()
        return tickers