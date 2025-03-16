import sqlite3
import os
from abc import ABC, abstractmethod
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
import time
from tqdm import tqdm


# ----------------------------------------------------------------
# 1) Datenbank-Verwaltung (Implementierung)
# ----------------------------------------------------------------

#region 1) Datenbank-Verwaltung (Implementierung)
class DatabaseType(Enum):
    SQLITE = 'sqlite'

def get_collection_database_repository(type: DatabaseType) -> dict:
    """
    Erstellt eine Sammlung von Datenbank-Repositories für verschiedene Underlying-Ticker.
    """
    if type == DatabaseType.SQLITE:
        # Alle Underlying-Ticker aus der .env lesen und bereinigen (Großbuchstaben)
        underlying_tickers = os.getenv('POLYGON_UNDERLYING_TICKER', 'UNKNOWN').replace(" ", "").upper().split(",")

        # Dictionary, das für jeden Ticker eine eigene Datenbank-Repository speichert
        repositories = {}

        for ticker in underlying_tickers:
            db_filename = f"data_collection/data/rawdata_{ticker.lower()}_db.sqlite"
            repositories[ticker] = SqliteDatabaseRepository(db_filename)

        return repositories
    else:
        raise ValueError("Invalid database type.")
#endregion

# ----------------------------------------------------------------
# 2) Abstrakte Klasse für Datenbank-Repositories
# ----------------------------------------------------------------

#region 2) Abstrakte Klasse für Datenbank-Repositories
class CollectionDatabaseRepository(ABC):
    """
    Abstrakte Klasse für eine Datenbank-Schnittstelle zur Speicherung von Finanzdaten.
    """

    @abstractmethod
    def collection_db_migrate(self):
        """Führt die initiale Migration der Datenbank durch (Tabellen erstellen)."""
        ...

    @abstractmethod
    def insert_contracts_bulk(self, contracts):
        """Führt einen Bulk-Insert für mehrere Verträge durch."""
        ...

    @abstractmethod
    def insert_contract_aggregates_bulk(self, aggregates):
        """Speichert eine große Menge an Aggregationsdaten in die Datenbank."""
        ...

    @abstractmethod
    def aggregate_exists(self, ticker, from_date, to_date):
        """Überprüft, ob Aggregationsdaten für einen bestimmten Zeitraum existieren."""
        ...

    @abstractmethod
    def insert_data_of_index(self, data, ticker):
        """Speichert Index-Daten wie Schlusskurse von YFinance."""
        ...

    @abstractmethod
    def insert_data_interest_rates(self, series_data, series_id):
        """Speichert Zinssatz-Daten aus FRED in der Datenbank."""
        ...

    @abstractmethod
    def insert_data_implied_volatility(self, ticker, data):
        """Speichert implizite Volatilitätsdaten in der Datenbank."""
        ...
#endregion

# ----------------------------------------------------------------
# 3) Implementierung der SQLite-Datenbank-Klasse
# ----------------------------------------------------------------

#region 3) Implementierung der SQLite-Datenbank-Klasse
class SqliteDatabaseRepository(CollectionDatabaseRepository):
    """
    SQLite-Datenbank-Repository zur Speicherung von Finanzdaten.
    """

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.connection = self.get_collection_database_connection()
        self.batch_size = 100000

    # ----------------------------------------------------------------
    # 3.1) Verbindung & Initialisierung
    # ----------------------------------------------------------------

    #region 3.1) Verbindung & Initialisierung
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(sqlite3.OperationalError),
        reraise=True
    )
    def get_collection_database_connection(self):
        """
        Erstellt eine Verbindung zur SQLite-Datenbank mit Optimierungen.
        """
        conn = sqlite3.connect(self.filename, isolation_level=None, check_same_thread=False)
        c = conn.cursor()
        c.execute("PRAGMA synchronous = OFF")
        c.execute("PRAGMA journal_mode = MEMORY")
        c.execute("PRAGMA cache_size = -64000")
        return conn

    def collection_db_migrate(self):
        """
        Erstellt notwendige Tabellen für die Speicherung von Options- und Marktdaten.
        """
        try:
            c = self.connection.cursor()

            # Erstelle die notwendigen Tabellen
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

            c.execute("""
                CREATE TABLE IF NOT EXISTS treasury_bill (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    interest_rate REAL,
                    series_id TEXT,
                    UNIQUE(date, series_id)
                )
            """)

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

            c.execute("""
                CREATE TABLE IF NOT EXISTS performance_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    date TEXT,
                    close REAL,
                    UNIQUE(ticker, date)
                )
            """)

            print("✅ Datenbankmigration abgeschlossen.")
        except sqlite3.Error as e:
            print(f"❌ Fehler während der Migration: {e}")

    def drop_tables(self, db_paths, tables_list):
        """
        Löscht angegebene Tabellen aus mehreren SQLite-Datenbanken.

        :param db_paths: Liste von Pfaden zu den SQLite-Datenbanken.
        :param tables_list: Liste der zu löschenden Tabellen.

        In der .env anzupassen:
        RUN_DROP_TABLES=
        TARGET_DB=
        TABLES_TO_DROP=
        """
        if not db_paths or not tables_list:
            print("⚠️ Keine Datenbanken oder Tabellen angegeben. Kein Löschvorgang erforderlich.")
            return

        try:
            for db_path in db_paths:
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                print(f"🔴 Verbindung zur Datenbank: {db_path}")

                for table in tables_list:
                    print(f"🗑 Lösche Tabelle `{table}` in `{db_path}`...")
                    c.execute(f"DROP TABLE IF EXISTS {table}")

                conn.commit()
                conn.close()
                print(f"✅ Tabellen {tables_list} wurden erfolgreich aus `{db_path}` gelöscht.")

        except sqlite3.Error as e:
            print(f"❌ Fehler beim Löschen der Tabellen: {e}")
    #endregion

    # ----------------------------------------------------------------
    # 3.2) Funktionen zur Speicherung von Kontrakten
    # ----------------------------------------------------------------

    #region 3.2) Funktionen zur Speicherung von Kontrakten
    def insert_contracts_bulk(self, contracts_list):
        """
        Optimierter Bulk-Insert für Optionsdaten in SQLite.
        - Nutzt große Batches
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

            for batch_start in tqdm(range(0, len(contracts_list), self.batch_size), desc="Batch-Insert in contracts",
                                    unit="batch"):
                batch = contracts_list[batch_start: batch_start + self.batch_size]

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
    #endregion

    # ----------------------------------------------------------------
    # 3.3) Funktionen zur Speicherung von Aggregationsdaten
    # ----------------------------------------------------------------

    #region 3.3) Funktionen zur Speicherung von Aggregationsdaten
    def insert_contract_aggregates_bulk(self, aggregates_list):
        """
        Optimierter Bulk-Insert für Contract-Aggregate-Daten in SQLite.
        - Nutzt große Batches (100.000)
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

            for batch_start in tqdm(range(0, len(aggregates_list), self.batch_size),
                                    desc="Batch-Insert in contract_aggregates",
                                    unit="batch"):
                batch = aggregates_list[batch_start: batch_start + self.batch_size]

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
        Überprüft, ob mindestens ein Eintrag in der Tabelle "contract_aggregates" vorhanden ist,
        für einen bestimmten "contract_ticker" und einen Zeitraum.
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
    #endregion

    # ----------------------------------------------------------------
    # 3.4) Zusätzliche Daten (Zinsen, Volatilität, Index)
    # ----------------------------------------------------------------

    #region 3.4) Zusätzliche Daten (Zinsen, Volatilität, Index)
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

    def insert_data_of_index(self, data, ticker):
        print(f"📊 Typ von `data` direkt nach Funktionsaufruf: {type(data)}")
        print(f"📊 Erste Zeichen von `data`, falls String: {data[:100] if isinstance(data, str) else 'Kein String'}")

        # ✅ Sicherstellen, dass `data` existiert und ein DataFrame ist
        if not isinstance(data, pd.DataFrame):
            print(f"❌ Fehler: `data` ist kein DataFrame! Stattdessen: {type(data)}")
            return

        # ✅ Falls `data` ein MultiIndex hat, entferne die zweite Ebene
        if isinstance(data.columns, pd.MultiIndex):
            print("⚠ MultiIndex erkannt – Entferne zweite Ebene...")
            data.columns = data.columns.droplevel(1)

        # ✅ Falls das Datum im Index steckt, als Spalte speichern
        if 'Date' not in data.columns and 'Datetime' not in data.columns:
            data = data.reset_index()

        # ✅ Falls `Datetime` vorhanden ist, benenne sie in `Date` um
        if 'Datetime' in data.columns:
            data.rename(columns={'Datetime': 'Date'}, inplace=True)

        # ✅ Sicherstellen, dass alle benötigten Spalten vorhanden sind
        required_columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            print(f"❌ Fehlende Spalten in `data`: {missing_columns}")
            return  # Falls Spalten fehlen, abbrechen

        # ✅ Debugging: Zeige die aktuellen Spaltennamen und erste Zeilen
        print("📊 Spalten nach Anpassung:", data.columns)
        print("📊 Erste Zeilen vor DB-Insert:")
        print(data.head())

        c = self.connection.cursor()

        for _, row in data.iterrows():
            try:
                # ✅ Sicherstellen, dass `Date` im richtigen Format ist
                date_str = row["Date"].strftime('%Y-%m-%d') if hasattr(row["Date"], 'strftime') else str(row["Date"])

                # ✅ Sichere Abfrage von Werten
                values = (
                    ticker,
                    date_str,
                    row.get("Close", None),
                    row.get("High", None),
                    row.get("Low", None),
                    row.get("Open", None),
                    row.get("Volume", None)
                )

                #print("📊 Einfügen in DB:", values)

                # ✅ SQL-Query für den Datenbank-Insert
                sql = ("INSERT INTO index_data (ticker, date, close, high, low, open, volume) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?)")

                c.execute(sql, values)
                self.connection.commit()
                print("✅ Daten erfolgreich eingefügt.")

            except Exception as e:
                #print(f"❌ Fehler beim Einfügen von {values}: {e}")
                self.connection.rollback()

        self.connection.commit()

    def insert_data_implied_volatility(self, data, ticker):
        print("Ticker type:", type(ticker), ticker)

        # Falls die Spalten ein MultiIndex sind, flache sie ab.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Falls "Datetime" vorhanden ist, verwende diese als Datumsspalte.
        if "Datetime" in data.columns:
            # Überschreibe ggf. vorhandene "Date"-Spalte, um sicherzustellen, dass das korrekte Datum genutzt wird.
            data["Date"] = data["Datetime"]

        # Wenn die Datumsspalte nicht vorhanden ist, in eine Spalte umwandeln.
        if 'Date' not in data.columns:
            data = data.reset_index()
            if 'Date' not in data.columns:
                data.rename(columns={"index": "Date"}, inplace=True)

        # Debug-Ausgabe: Zeige die Spaltennamen
        print("Spalten nach reset_index:", data.columns)

        c = self.connection.cursor()

        for _, row in data.iterrows():
            # Nutze den Wert aus der "Date"-Spalte, der nun das richtige Datum (aus "Datetime") enthält.
            date_value = row["Date"]
            if hasattr(date_value, 'strftime'):
                date_str = date_value.strftime('%Y-%m-%d')
            else:
                date_str = str(date_value)

            # Zugriff auf die einzelnen Spaltenwerte als skalare Werte
            close_val = row["Close"]
            high_val = row["High"]
            low_val = row["Low"]
            open_val = row["Open"]
            volume_val = row["Volume"]

            values = (
                ticker,  # Ticker (String)
                date_str,  # Datum als String
                close_val,  # Schlusskurs
                high_val,  # Tageshoch
                low_val,  # Tagestief
                open_val,  # Eröffnungskurs
                volume_val  # Handelsvolumen
            )

            print("Einfügen:", values)

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

    def insert_data_performance_index(self, data, ticker):
        # Falls die Spalten ein MultiIndex sind, flache sie ab.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Wenn "Datetime" vorhanden ist, verwende diese Spalte als Datum.
        if "Datetime" in data.columns:
            data["Date"] = data["Datetime"]

        # Falls "Date" nicht vorhanden ist, den Index zurücksetzen und ggf. umbenennen.
        if "Date" not in data.columns:
            data = data.reset_index()
            if "Date" not in data.columns:
                data.rename(columns={"index": "Date"}, inplace=True)

        # Debug-Ausgabe: Zeige die Spaltennamen
        print("Spalten nach reset_index:", data.columns)

        c = self.connection.cursor()

        for _, row in data.iterrows():
            # Datum extrahieren – benutze "Date" (das ggf. aus "Datetime" stammt)
            date_value = row["Date"]
            if hasattr(date_value, 'strftime'):
                date_str = date_value.strftime('%Y-%m-%d')
            else:
                date_str = str(date_value)

            # Hole den Schlusskurs
            close_val = row["Close"]

            values = (ticker, date_str, close_val)
            print("Einfügen:", values)

            sql = ("INSERT INTO performance_index (ticker, date, close) VALUES (?, ?, ?)")
            try:
                c.execute(sql, values)
                self.connection.commit()
                print("Daten erfolgreich eingefügt.")
            except Exception as e:
                print("Fehler beim Einfügen der Daten:", e)
                self.connection.rollback()

        self.connection.commit()

    #endregion

    # ----------------------------------------------------------------
    # 3.5) Helfer-Funktionen
    # ----------------------------------------------------------------

    #region 3.5) Helfer-Funktion
    def count_contracts(self):
        """
        Zählt die Anzahl der Einträge in der Contracts-Tabelle.
        """
        query = "SELECT COUNT(*) FROM contracts"
        c = self.connection.cursor()
        c.execute(query)
        count = c.fetchone()[0]
        return count

    def count_aggregates(self):
        """
        Zählt die Anzahl der Einträge in der contract_aggregates-Tabelle.
        """
        query = "SELECT COUNT(*) FROM contract_aggregates"
        c = self.connection.cursor()
        c.execute(query)
        count = c.fetchone()[0]
        return count
    #endregion
#endregion
