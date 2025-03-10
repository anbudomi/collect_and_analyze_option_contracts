import sqlite3
import os
from datetime import datetime, timedelta
import numpy as np
import tqdm
import scipy.interpolate
import re
from collections import defaultdict


class DataPreparer:
    def __init__(self, raw_database_path, prepared_db_path, index, index_ticker):
        self.raw_db_path = os.path.join(raw_database_path, f"rawdata_{index.lower()}_db.sqlite")
        self.prepared_db_path = prepared_db_path
        self.index = index
        self.index_ticker = index_ticker
        self.batch_size = 50000

    def process_and_filter_contracts(self, database_repository):
        if not os.path.exists(self.raw_db_path):
            print(f"❌ Fehler: Rohdatenbank {self.raw_db_path} existiert nicht!")
            return

        print(f"🔄 Verarbeitung von {self.raw_db_path} gestartet...")

        try:
            raw_conn = sqlite3.connect(self.raw_db_path)
            raw_cursor = raw_conn.cursor()

            raw_cursor.execute("""
                SELECT c.ticker, c.date, c.strike_price, c.expiration_date, ca.c
                FROM contracts c
                INNER JOIN contract_aggregates ca 
                ON c.ticker = ca.contract_ticker AND c.date = ca.date
            """)
            contracts = raw_cursor.fetchall()
            print(f"✅ {len(contracts)} relevante Contracts gefunden.")
        except sqlite3.Error as e:
            print(f"⚠ Fehler beim Laden der Contracts: {e}")
            return

        prepared_data = []
        date_format = "%Y-%m-%d"

        for ticker, date, execution_price, expiration_date, market_price_option in tqdm.tqdm(contracts, desc=f"Verarbeite {self.index}"):
            try:
                date_obj = datetime.strptime(date, date_format)
                expiration_date_obj = datetime.strptime(expiration_date, date_format)
                remaining_time = (expiration_date_obj - date_obj).days

                if remaining_time == 0:
                    continue

                raw_cursor.execute("SELECT close FROM index_data WHERE date = ?", (date,))
                market_price_base = raw_cursor.fetchone()
                market_price_base = market_price_base[0] if market_price_base else None

                raw_cursor.execute("SELECT close FROM implied_volatility WHERE date = ?", (date,))
                implied_volatility = raw_cursor.fetchone()
                implied_volatility = implied_volatility[0] if implied_volatility else None

                tb_id_list = self.get_treasury_bill_id(remaining_time)
                tb_days_list = [self.get_days_from_id(id) for id in tb_id_list]
                tb_rates_list = [self.get_treasury_rate(raw_cursor, date, id) for id in tb_id_list]

                risk_free_rate = max(tb_rates_list[0], 0)
                if len(tb_id_list) > 1 and len(tb_rates_list) > 1:
                    interpolation_fct = self.log_linear_interest_interpolation(tb_rates_list, tb_days_list)
                    risk_free_rate = round(interpolation_fct(remaining_time), 4)

                prepared_data.append((ticker, execution_price, market_price_base, remaining_time,
                                      risk_free_rate, date, market_price_option, implied_volatility))
            except sqlite3.Error as e:
                print(f"⚠ Fehler bei Datensatz {ticker}: {e}")
                continue

        print(f"🔍 {len(prepared_data)} Verträge vor der Call-Put-Filterung.")
        prepared_data = self.filter_valid_option_pairs(prepared_data)
        print(f"✅ {len(prepared_data)} gültige Call-Put-Paare nach Filterung.")

        # Batch-Verarbeitung nach der Filterung
        for i in range(0, len(prepared_data), self.batch_size):
            batch = prepared_data[i:i + self.batch_size]
            print(f"📥 Speichere {len(batch)} Einträge...")
            database_repository.bulk_insert(batch, self.index)

        raw_conn.close()
        print("✅ Verarbeitung abgeschlossen.")

    def filter_valid_option_pairs(self, prepared_data):
        """Filtert gültige Call-Put-Paare basierend auf der Call-Put-Parität."""
        filtered_data = []
        ticker_map = defaultdict(dict)

        def extract_base_ticker(ticker):
            """Extrahiert den Basisticker (ohne letztes C/P)."""
            match = re.match(r"(.*?)(C|P)(\d+)$", ticker)
            return match.groups() if match else (ticker, None)

        for data in prepared_data:
            ticker, execution_price, market_price_base, remaining_time, risk_free_rate, date, market_price_option, implied_volatility = data
            base_ticker, option_type, strike_price = extract_base_ticker(ticker)
            print(f"📌 Parsed: {ticker} → Base: {base_ticker}, Type: {option_type}, Strike: {strike_price}")
            if option_type:
                ticker_map[base_ticker][option_type] = data

        for base_ticker, options in ticker_map.items():
            if 'C' in options and 'P' in options:
                call_data = options['C']
                put_data = options['P']
                put_price = put_data[6]
                market_price_base = put_data[2]
                remaining_time = put_data[3] / 365
                risk_free_rate = put_data[4]
                calc_call = put_price + market_price_base - remaining_time * np.exp(-risk_free_rate * remaining_time)
                actual_call_price = call_data[6]
                deviation = abs(calc_call - actual_call_price) / actual_call_price
                print(f"🔢 Base: {base_ticker} | Calc: {calc_call:.4f} | Actual: {actual_call_price:.4f} | Deviation: {deviation:.2%}")
                if deviation <= 0.5:
                    filtered_data.append(call_data)
                    filtered_data.append(put_data)

        return filtered_data



    def get_treasury_rate(self, cursor, date, series_id):
        cursor.execute("SELECT interest_rate FROM treasury_bill WHERE date = ? AND series_id = ?", (date, series_id))
        result = cursor.fetchone()

        if result[0] is not None:
            return round(result[0] / 100, 4)

        # Falls keine Daten vorhanden, Suche in beiden Richtungen bis zu 10 Tage
        max_days_back = 10
        date_obj = datetime.strptime(date, "%Y-%m-%d")

        for offset in range(1, max_days_back + 1):
            prev_date = (date_obj - timedelta(days=offset)).strftime("%Y-%m-%d")
            next_date = (date_obj + timedelta(days=offset)).strftime("%Y-%m-%d")

            cursor.execute("SELECT interest_rate FROM treasury_bill WHERE date = ? AND series_id = ?",
                           (prev_date, series_id))
            result = cursor.fetchone()
            if result is not None:
                return round(result[0] / 100, 4)

            cursor.execute("SELECT interest_rate FROM treasury_bill WHERE date = ? AND series_id = ?",
                           (next_date, series_id))
            result = cursor.fetchone()
            if result is not None:
                return round(result[0] / 100, 4)

        print(f"⚠ Keine Zinsdaten für {date}, setze 0.0.")
        return 0.0

    def log_linear_interest_interpolation(self, rates, days):
        if any(x <= 0 for x in rates):
            rates = [max(x, 0.0001) for x in rates]
        log_days = np.log10(days)
        log_rates = np.log10(rates)
        lin_interp = scipy.interpolate.interp1d(log_days, log_rates, kind='linear', fill_value="extrapolate")
        return lambda target_days: np.power(10.0, lin_interp(np.log10(target_days)))

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