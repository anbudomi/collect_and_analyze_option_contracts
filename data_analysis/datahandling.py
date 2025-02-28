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