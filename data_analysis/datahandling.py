import sqlite3
import os
import numpy as np
import scipy.interpolate as interp
from collections import defaultdict
from datetime import datetime, timedelta
from tqdm import tqdm
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy
from scipy.stats import norm
import time
import yfinance as yf

class DataHandler:
    def __init__(
            self,
            raw_database_path,
            raw_db_names,
            sorted_db_path,
            batch_size
    ):
        self.raw_database_path = raw_database_path
        self.raw_db_names = raw_db_names
        self.sorted_db_path = sorted_db_path
        self.batch_size = batch_size

        # Dynamische Erstellung der Tabellennamen basierend auf den Indexnamen
        self.sorted_table_names = [f"sorted_{self.extract_index_name(db_name)}_data" for db_name in self.raw_db_names]

    def extract_index_name(self, db_name):
        """Extrahiert den Indexnamen aus dem Datenbanknamen (z. B. 'ndx' oder 'spx')."""
        match = re.search(r"data_(.*?)_", db_name)
        return match.group(1) if match else "unknown"

    def log_linear_interest_interpolation(self, rates, days):
        if len(rates) > 1 and len(days) > 1:
            if any(x <= 0 for x in rates):
                rates = self.check_rates_for_zeros(rates)
            log_days = np.log10(days)
            log_rates = np.log10(rates)

            lin_interp = scipy.interpolate.interp1d(log_days, log_rates, kind='linear', fill_value="extrapolate")
            log_interp = lambda target_days: np.power(10.0, lin_interp(np.log10(target_days)))

            return log_interp


    def check_rates_for_zeros(self, rates):
        theoretical_zero = 1e-16
        for i in range(len(rates)):
            if rates[i] <= 0:
                rates[i] = theoretical_zero
        return rates



    def process_contracts(self, database_repository):
        """Liest nur relevante Contracts aus und bereitet sie für den Bulk-Insert vor."""
        raw_dbs = [os.path.join(self.raw_database_path, db_name) for db_name in self.raw_db_names]
        print("Starte Verarbeitung der Contracts...")

        for db_path, table_name in zip(raw_dbs, self.sorted_table_names):
            print(f"Verarbeite Datenbank: {db_path.replace(os.sep, '/')}, Ziel-Tabelle: {table_name}")

            if not os.path.exists(db_path):
                print(f"Warnung: Rohdatenbank {db_path} existiert nicht!")
                continue

            try:
                raw_conn = sqlite3.connect(db_path)
                raw_cursor = raw_conn.cursor()

                # Nur Contracts mit vorhandenem Eintrag in contract_aggregates laden
                raw_cursor.execute("""
                    SELECT c.ticker, c.date, c.strike_price, c.expiration_date
                    FROM contracts c
                    INNER JOIN contract_aggregates ca 
                    ON c.ticker = ca.contract_ticker AND c.date = ca.date
                """)
                contracts = raw_cursor.fetchall()
                print(f"{len(contracts)} relevante Contracts in {db_path.replace(os.sep, '/')} gefunden.")

            except sqlite3.Error as e:
                print(f"Fehler beim Zugriff auf {db_path}: {e}")
                continue

            prepared_data = []
            total_contracts = len(contracts)
            date_format = "%Y-%m-%d"

            for i, (ticker, date, execution_price, expiration_date) in enumerate(
                    tqdm(contracts, desc=f"Verarbeite {table_name}"), start=1):
                try:
                    date_obj = datetime.strptime(date, date_format)
                    expiration_date_obj = datetime.strptime(expiration_date, date_format)
                    remaining_time = int((expiration_date_obj - date_obj).days)

                    # Ausschluss von Einträgen mit remaining_time == 0
                    if remaining_time == 0:
                        continue  # Überspringt diesen Datensatz

                    raw_cursor.execute("SELECT close FROM index_data WHERE date = ?", (date,))
                    market_price_base = raw_cursor.fetchone()
                    market_price_base = market_price_base[0] if market_price_base else None

                    raw_cursor.execute("SELECT c FROM contract_aggregates WHERE contract_ticker = ? AND date = ?",
                                       (ticker, date))
                    market_price_option = raw_cursor.fetchone()
                    market_price_option = market_price_option[0] if market_price_option else None

                    raw_cursor.execute("SELECT close FROM implied_volatility WHERE date = ?", (date,))
                    implied_volatility = raw_cursor.fetchone()
                    implied_volatility = implied_volatility[0] if implied_volatility else None

                    #def get_treasury_bill_id(self, remaining_time):
                    #def get_days_from_id(self, id_string):
                    tb_id_list = self.get_treasury_bill_id(remaining_time)
                    tb_days_list = []
                    for id in tb_id_list:
                        tb_days_list.append(self.get_days_from_id(id))
                    tb_rates_list = []
                    for id in tb_id_list:
                        # Suche nach dem letzten bekannten Zinssatz
                        raw_cursor.execute("""
                            SELECT interest_rate FROM treasury_bill
                            WHERE date = ? AND series_id = ?
                        """, (date, id))
                        result = raw_cursor.fetchone()
                        interest_rate = result[0] if result else None

                        # Falls kein Wert gefunden wurde, gehe maximal 10 Tage zurück
                        max_days_back = 10
                        days_back = 0
                        temp_date = date_obj - timedelta(days=1)

                        while interest_rate is None and days_back < max_days_back:
                            temp_date_str = temp_date.strftime(date_format)
                            raw_cursor.execute("""
                                SELECT interest_rate FROM treasury_bill
                                WHERE date = ? AND series_id = ?
                            """, (temp_date_str, id))
                            result = raw_cursor.fetchone()

                            if result:
                                interest_rate = result[0]
                            else:
                                temp_date -= timedelta(days=1)  # Rückwärtsgehen
                                days_back += 1  # Zähle die Tage zurück

                        # Falls nach 10 Tagen kein Zinssatz gefunden wurde, setze Standardwert
                        if interest_rate is None:
                            print(
                                f"⚠ Keine Zinsdaten für {date} innerhalb von {max_days_back} Tagen. Setze Standardwert.")
                            interest_rate = 0.0  # Oder ein sinnvoller Wert (z.B. Durchschnitt)

                        tb_rates_list.append(round(interest_rate/100, 4))

                    if len (tb_id_list) > 1 and len(tb_rates_list) > 1:
                        interpolation_fct = self.log_linear_interest_interpolation(tb_rates_list, tb_days_list)
                        risk_free_rate = round(interpolation_fct(remaining_time), 4)
                    else:
                        if tb_rates_list[0] < 0:
                            risk_free_rate = 0
                        else:
                            risk_free_rate = tb_rates_list[0]

                    prepared_data.append((ticker, execution_price, market_price_base, remaining_time,
                                          risk_free_rate, date, market_price_option, implied_volatility))

                    if i % self.batch_size == 0 or i == total_contracts:
                        print(f"Starte optimierten Bulk-Insert für {len(prepared_data)} Einträge in {table_name}...")
                        database_repository.bulk_insert(prepared_data, table_name)
                        print(f"{i}/{total_contracts} Verträge verarbeitet und gespeichert in {table_name}")
                        prepared_data = []  # Liste zurücksetzen
                except sqlite3.Error as e:
                    print(f"Fehler beim Verarbeiten von Datensatz {i} in {db_path}: {e}")
                    continue

            raw_conn.close()

        print("Alle relevanten Contracts verarbeitet.")

    def get_days_from_id(self, id_string):
        if id_string == "DTB4WK":
            return 28
        if id_string == "DTB3":
            return 84
        if id_string == "DTB6":
            return 182
        if id_string == "DTB1YR":
            return 364

    def get_treasury_bill_id(self, remaining_time):
        if remaining_time <= 28:
            id_list = ['DTB4WK']
            return id_list
        if remaining_time > 364:
            id_list = ['DTB1YR']
            return id_list
        if remaining_time > 28 and remaining_time <= 84:
            id_list = ['DTB4WK', 'DTB3']
            return id_list
        if remaining_time > 84 and remaining_time <= 182:
            id_list = ['DTB3', 'DTB6']
            return id_list
        if remaining_time > 182 and remaining_time <= 364:
            id_list = ['DTB6', 'DTB1YR']
            return id_list


class DataPreparer:
    def __init__(
            self,
            sorted_db_path,
            prepared_db_path,
            indices_to_prepare,
            index
    ):
        self.sorted_db_path = sorted_db_path
        self.prepared_db_path = prepared_db_path
        self.indices_to_prepare = indices_to_prepare
        self.sorted_table_name = f"sorted_{index.lower()}_data"
        self.prepared_table_name = f"prepared_{index.lower()}_data"
        self.index = index

        self.initialize_prepared_db()

    def initialize_prepared_db(self):
        """Kopiert die sortierte Datenbank, falls sie noch nicht existiert und erstellt die vorbereitete Tabelle."""
        if not os.path.exists(self.prepared_db_path):
            conn_prepared = sqlite3.connect(self.prepared_db_path)
            conn_prepared.close()
            print("✅ `prepared_data_db.sqlite` wurde erstellt.")

        conn_sorted = sqlite3.connect(self.sorted_db_path)
        conn_prepared = sqlite3.connect(self.prepared_db_path)
        cursor_sorted = conn_sorted.cursor()

        # Prüfen, ob die Tabelle existiert
        cursor_sorted.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor_sorted.fetchall()]

        if self.sorted_table_name in tables:
            df = pd.read_sql(f"SELECT * FROM {self.sorted_table_name}", conn_sorted)

            # ❗ Falls `prepared_ndx_data` bereits existiert, NICHT überschreiben, sondern nur falls sie leer ist
            existing_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn_prepared)

            if self.prepared_table_name in existing_tables["name"].values:
                print(f"⚡ Tabelle `{self.prepared_table_name}` existiert bereits. Keine erneute Kopie nötig.")
            else:
                df.to_sql(self.prepared_table_name, conn_prepared, if_exists='replace', index=False)
                print(
                    f"✅ Tabelle {self.sorted_table_name} wurde als {self.prepared_table_name} in `prepared_data_db.sqlite` kopiert.")

        else:
            print(
                f"⚠ Tabelle {self.sorted_table_name} nicht gefunden. Stelle sicher, dass die Sorted-DB korrekt erstellt wurde!")

        conn_sorted.close()
        conn_prepared.close()

    def convert_implied_volatility(self):
        """Fügt die Spalte `implied_volatility[dec]` hinzu und befüllt sie, falls noch nicht vorhanden."""
        conn = sqlite3.connect(self.prepared_db_path)
        cursor = conn.cursor()

        # Prüfen, ob die Spalte existiert
        cursor.execute(f"PRAGMA table_info({self.prepared_table_name})")
        columns = [col[1] for col in cursor.fetchall()]

        if "implied_volatility[dec]" not in columns:
            cursor.execute(f"ALTER TABLE {self.prepared_table_name} ADD COLUMN `implied_volatility[dec]` REAL")
            conn.commit()

        # Update nur für Zeilen, wo die Spalte NULL ist (damit nichts überschrieben wird)
        cursor.execute(f"""
            UPDATE {self.prepared_table_name}
            SET `implied_volatility[dec]` = ROUND(implied_volatility / 100, 4)
            WHERE `implied_volatility[dec]` IS NULL
        """)

        conn.commit()
        conn.close()
        print(f"✅ `implied_volatility[dec]` erfolgreich konvertiert und gespeichert für {self.prepared_table_name}.")

    def calc_dividend_yield(self):
        """Fügt die Spalte `dividend_yield` hinzu und berechnet sie, falls noch nicht vorhanden."""
        conn = sqlite3.connect(self.prepared_db_path)
        cursor = conn.cursor()

        # Prüfen, ob die Spalte existiert
        cursor.execute(f"PRAGMA table_info({self.prepared_table_name})")
        columns = [col[1] for col in cursor.fetchall()]

        if "dividend_yield" not in columns:
            cursor.execute(f"ALTER TABLE {self.prepared_table_name} ADD COLUMN `dividend_yield` REAL")
            conn.commit()

        # Alle einzigartigen Handelsdaten holen
        cursor.execute(f"SELECT DISTINCT trade_date FROM {self.prepared_table_name}")
        trade_dates = [row[0] for row in cursor.fetchall()]

        # Lade Closing Price Daten aus `closing_candles_db`
        closing_db_path = os.path.join(os.path.dirname(self.prepared_db_path), "closing_candles_db.sqlite")
        conn_closing = sqlite3.connect(closing_db_path)
        closing_data = pd.read_sql(f"SELECT date, close FROM closing_data_{self.index.lower()}", conn_closing)
        conn_closing.close()

        # Falls Closing Price-Daten nicht verfügbar sind
        if closing_data.empty:
            print(f"⚠ Keine Closing Price-Daten für {self.index} gefunden. Überspringe Dividend Yield-Berechnung.")
            return

        closing_data['date'] = pd.to_datetime(closing_data['date'])

        # Berechnung des Dividend Yields für jedes einzigartige Handelsdatum
        for trade_date in trade_dates:
            trade_date_dt = pd.to_datetime(trade_date)
            dividend_yield = self.calculate_dividend_yield(trade_date_dt, closing_data)

            if dividend_yield is not None:
                cursor.execute(f"""
                    UPDATE {self.prepared_table_name}
                    SET dividend_yield = ?
                    WHERE trade_date = ? AND dividend_yield IS NULL
                """, (dividend_yield, trade_date))

        conn.commit()
        conn.close()
        print(f"✅ `dividend_yield` erfolgreich berechnet und gespeichert für {self.prepared_table_name}.")

    def calculate_dividend_yield(self, trade_date, index_data_df):
        """
        Berechnet die Dividendenrendite für ein bestimmtes Datum.
        Holt Closing Prices jetzt aus `closing_candles_db.sqlite`.
        """
        one_year_ago = trade_date - timedelta(days=365)

        # Suche das nächstgelegene Datum (Trade Date & ein Jahr zuvor)
        nearest_date_t = self.find_nearest_previous_date(trade_date, index_data_df['date'])
        nearest_date_t1y = self.find_nearest_previous_date(one_year_ago, index_data_df['date'])

        if nearest_date_t is None or nearest_date_t1y is None:
            print(f"⚠ Kein gültiges Datum für {trade_date.strftime('%Y-%m-%d')} gefunden!")
            return None

        # Preise aus `closing_candles_db.sqlite` holen
        price_index_t = index_data_df.loc[index_data_df['date'] == nearest_date_t, 'close'].values
        price_index_t1y = index_data_df.loc[index_data_df['date'] == nearest_date_t1y, 'close'].values

        if price_index_t.size == 0 or price_index_t1y.size == 0:
            print(
                f"⚠ Kein Closing-Preis für {nearest_date_t.strftime('%Y-%m-%d')} oder {nearest_date_t1y.strftime('%Y-%m-%d')} gefunden!")
            return None

        # Berechnung der Dividendenrendite
        price_index_t = price_index_t[0]
        price_index_t1y = price_index_t1y[0]

        return (price_index_t / price_index_t1y) - 1  # (P_t / P_t-1y) - 1

    def find_nearest_previous_date(self, target_date, date_series):
        """
        Findet das nächstgelegene vorherige Datum in einer Datumsreihe.
        """
        valid_dates = date_series[date_series <= target_date]
        if valid_dates.empty:
            return None
        return valid_dates.max()

    def fetch_yfinance_data(self, ticker, start_date, end_date):
        """
        Holt Closing-Price-Daten von Yahoo Finance und speichert sie direkt in `closing_candles_db.sqlite`.
        """
        closing_db_path = os.path.join(os.path.dirname(self.prepared_db_path), "closing_candles_db.sqlite")
        closing_table_name = f"closing_data_{self.index.lower()}"

        max_retries = 5
        wait_time = 10  # Startwartezeit in Sekunden

        for attempt in range(max_retries):
            try:
                print(f"📊 Fetching Closing Prices for {ticker}... (Try {attempt + 1}/{max_retries})")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if not isinstance(data, pd.DataFrame) or data.empty:
                    print(f"⚠️ Keine Daten für {ticker} erhalten. Erneuter Versuch...")
                    raise ValueError("Empty Data")

                # ✅ Daten formatieren
                formatted_data = self.format_yfinance_data(data)

                if formatted_data.empty:
                    print(f"⚠️ Keine formatierbaren Daten für {ticker} erhalten. Erneuter Versuch...")
                    raise ValueError("Formatted Data Empty")

                # 🔹 **Datum korrekt als YYYY-MM-DD formatieren**
                formatted_data['Datetime'] = formatted_data['Datetime'].dt.strftime('%Y-%m-%d')

                # ✅ Speichern in SQLite-Datenbank
                conn = sqlite3.connect(closing_db_path)
                cursor = conn.cursor()

                # Tabelle erstellen, falls sie nicht existiert
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {closing_table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT UNIQUE,
                        close REAL
                    )
                """)

                # Daten in die DB speichern (ersetze vorhandene Einträge für den Tag)
                formatted_data = formatted_data[['Datetime', 'Close']].rename(
                    columns={"Datetime": "date", "Close": "close"})

                formatted_data.to_sql(closing_table_name, conn, if_exists='replace', index=False)
                conn.commit()
                conn.close()

                print(
                    f"✅ Closing-Price-Daten für {ticker} gespeichert in `{closing_db_path}` (Tabelle `{closing_table_name}`).")
                return

            except Exception as e:
                if "Too Many Requests" in str(e) or "Empty Data" in str(e):
                    print(f"⏳ Warte {wait_time} Sekunden wegen Rate-Limit oder leerer Daten...")
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponentielles Warten
                else:
                    print(f"❌ Fehler für {ticker}: {e}")
                    break  # Kein erneuter Versuch bei anderen Fehlern

    #Der alte Stand hat ohne diese Funktion funktioniert, allerdings hat YahooFinance ihre API verändert
    def format_yfinance_data(self, input_df):
        """
        Formatiert die von Yahoo Finance abgerufenen Daten in das benötigte Format.
        """
        # ✅ Falls MultiIndex vorhanden ist, entferne die erste Ebene ("Price")
        if isinstance(input_df.columns, pd.MultiIndex):
            input_df.columns = input_df.columns.droplevel(1)

        # ✅ Index zurücksetzen, damit das Datum eine Spalte wird
        input_df = input_df.reset_index()

        # ✅ Spaltennamen normalisieren (yfinance verwendet manchmal "Date" statt "Datetime")
        if "Date" in input_df.columns:
            input_df.rename(columns={"Date": "Datetime"}, inplace=True)

        # ✅ Nur relevante Spalten behalten
        required_columns = ["Datetime", "Close"]
        available_columns = [col for col in required_columns if col in input_df.columns]

        return input_df[available_columns]


class DataAnalyzer:
    def __init__(
            self,
            sorted_db_path,
            indices_to_analyze
    ):
        self.sorted_db_path = sorted_db_path
        self.indices_to_analyze = indices_to_analyze

    def load_data(self, index_name):
        """Lädt Optionsdaten für einen bestimmten Index aus der SQLite-Datenbank"""
        table_name = f"sorted_{index_name.lower()}_data"  # Dynamischer Tabellenname
        conn = sqlite3.connect(self.sorted_db_path)

        query = f"""
        SELECT trade_date, execution_price AS strike_price, 
               market_price_base AS spot_price, ticker
        FROM {table_name}
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # Sicherstellen, dass trade_date als Datum erkannt wird
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['month'] = df['trade_date'].dt.to_period('M')  # Monat extrahieren
        return df

    def classify_options(self, df):
        """Berechnet ITM/OTM Status basierend auf Ticker (Call/Put Unterscheidung)"""

        def classify(row):
            ticker = row['ticker'].lower()
            if "c" in ticker:  # Annahme: 'C' für Call, 'P' für Put
                return 'ITM' if row['spot_price'] > row['strike_price'] else 'OTM'
            elif "p" in ticker:
                return 'ITM' if row['spot_price'] < row['strike_price'] else 'OTM'
            else:
                return 'Unknown'

        df['status'] = df.apply(classify, axis=1)
        return df


    def analyze_and_plot(self):
        """Erstellt Diagramme für die ITM/OTM-Verteilung - Gesamt & getrennt nach Call/Put für SPX & NDX"""
        for index in self.indices_to_analyze:
            df = self.load_data(index)
            df = self.classify_options(df)

            # 1) Gesamtübersicht für den Index (wie bisher)
            self.plot_itm_otm_distribution(df, index, f'ITM vs. OTM Verteilung für {index}')

            # 2) Separate Analysen für Call und Put Optionen
            df_call = df[df['ticker'].str.contains(r'C\d{6}', regex=True, case=False)]
            df_put = df[df['ticker'].str.contains(r'P\d{6}', regex=True, case=False)]

            if not df_call.empty:
                self.plot_itm_otm_distribution(df_call, index, f'ITM vs. OTM {index} Call')
            if not df_put.empty:
                self.plot_itm_otm_distribution(df_put, index, f'ITM vs. OTM {index} Put')

    def plot_itm_otm_distribution(self, df, index, title):
        """Erstellt ein gestapeltes Balkendiagramm für die ITM/OTM-Verteilung"""
        monthly_counts = df.groupby(['month', 'status']).size().unstack().fillna(0)
        monthly_percent = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100

        # Plot erstellen
        fig, ax = plt.subplots(figsize=(14, 6))
        monthly_percent.plot(kind='bar', stacked=True, alpha=0.85, ax=ax)

        # Titel & Achsen
        ax.set_title(title)
        ax.set_xlabel('Monat')
        ax.set_ylabel('Prozent')
        ax.legend(title='Status')

        # X-Achse optimieren
        ax.set_xticks(range(0, len(monthly_percent.index), max(1, len(monthly_percent.index) // 12)))
        ax.set_xticklabels(monthly_percent.index[::max(1, len(monthly_percent.index) // 12)], rotation=45, ha="right")

        # Gitterlinien für bessere Lesbarkeit
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Diagramm anzeigen
        plt.show()