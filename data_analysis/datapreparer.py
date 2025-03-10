import os
import sqlite3
import math
import datetime
from datetime import timedelta

# Neue Version der DataPreparer-Klasse, die die Daten in die prefiltered_DB schreibt.
class DataPreparer:
    def __init__(self, raw_database_path, prefiltered_db_path, prepared_db_path, index, index_ticker):
        # Der Pfad zur Rohdatenbank wird anhand des Index konstruiert.
        self.raw_db_path = os.path.join(raw_database_path, f"rawdata_{index.lower()}_db.sqlite")
        self.prefiltered_db_path = prefiltered_db_path  # Pfad zur prefiltered_DB
        self.prepared_db_path = prepared_db_path
        self.index = index
        self.index_ticker = index_ticker
        self.batch_size = 50000

    def compute_risk_free_rate(self, cursor, date_str, remaining_days):
        """
        Berechnet den risikofreien Zinssatz anhand der treasury_bill-Daten und interpoliert,
        falls zwei unterschiedliche Serien (Laufzeiten) vorliegen.
        """
        series_ids = self.get_treasury_bill_id(remaining_days)
        if not series_ids:
            return 0.01
        elif len(series_ids) == 1:
            return self.get_treasury_rate(cursor, date_str, series_ids[0])
        elif len(series_ids) == 2:
            r1 = self.get_treasury_rate(cursor, date_str, series_ids[0])
            r2 = self.get_treasury_rate(cursor, date_str, series_ids[1])
            d1 = self.get_days_from_id(series_ids[0])
            d2 = self.get_days_from_id(series_ids[1])
            interp_func = self.log_linear_interest_interpolation([r1, r2], [d1, d2])
            rate = round(interp_func(remaining_days), 4)
            return rate
        else:
            return 0.01  # Fallback

    def process_prefiltered_data(self, prefiltered_repo):
        """
        Liest die Rohdaten, führt den Join zwischen contract_aggregates und contracts
        sowie weitere Abfragen durch und fügt die Daten in die prefiltered_DB ein.

        Dabei wird:
         - Aus contract_aggregates: (contract_ticker, date, c) als market_price_option geladen.
         - Aus contracts: underlying_ticker, contract_type, expiration_date, strike_price
           anhand von ticker (gleich contract_ticker) und date abgefragt.
         - Aus implied_volatility: der close-Wert (als implizite Volatilität) für den entsprechenden date.
         - Aus index_data: der close-Wert (als market_price_base) für den entsprechenden date.
         - Und aus treasury_bill: der risk_free_rate über Interpolation.
        """
        # Verbindung zur Rohdatenbank herstellen
        raw_conn = sqlite3.connect(self.raw_db_path)
        raw_cursor = raw_conn.cursor()

        # Lade alle Datensätze aus contract_aggregates: (contract_ticker, date, c)
        raw_cursor.execute("SELECT contract_ticker, date, c FROM contract_aggregates")
        ca_data = raw_cursor.fetchall()

        # Lade implizite Volatilitätsdaten: Mapping: date -> close
        raw_cursor.execute("SELECT date, close FROM implied_volatility")
        iv_data = {row[0]: row[1] for row in raw_cursor.fetchall()}

        # Lade Indexdaten: Mapping: date -> close
        raw_cursor.execute("SELECT date, close FROM index_data")
        index_data = {row[0]: row[1] for row in raw_cursor.fetchall()}

        valid_rows = []
        for ca in ca_data:
            contract_ticker, date_str, market_price_option = ca

            # Finde passenden Eintrag in contracts anhand von contract_ticker und date.
            # Hier holen wir auch strike_price, der als execution_price verwendet werden soll.
            raw_cursor.execute("""
                SELECT underlying_ticker, contract_type, expiration_date, strike_price
                FROM contracts
                WHERE ticker = ? AND date = ?
                LIMIT 1
            """, (contract_ticker, date_str))
            contract_row = raw_cursor.fetchone()
            if contract_row is None:
                continue

            underlying_ticker, contract_type, expiration_date_str, strike_price = contract_row

            try:
                expiration_date = datetime.datetime.strptime(expiration_date_str, "%Y-%m-%d")
                trade_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                continue

            remaining_days = (expiration_date - trade_date).days
            if remaining_days <= 0:
                continue
            remaining_time = round(remaining_days / 365.0, 4)

            # Marktpreis-Basis aus index_data: runde auf 2 Dezimalstellen
            if date_str not in index_data:
                continue
            market_price_base = round(index_data[date_str], 2)

            # Implicit Volatility: Hole den close-Wert aus iv_data und runde auf 4 Nachkommastellen.
            if date_str not in iv_data:
                continue
            imp_vol_percent = round(iv_data[date_str], 4)
            imp_vol_dec = round(imp_vol_percent / 100.0, 4)

            # Risk-free rate aus treasury_bill berechnen (über Interpolation)
            risk_free_rate = self.compute_risk_free_rate(raw_cursor, date_str, remaining_days)

            # Hier: execution_price wird aus contracts (strike_price) genommen.
            data_tuple = (
                contract_ticker,
                underlying_ticker,
                date_str,
                contract_type,
                expiration_date_str,
                remaining_days,
                remaining_time,
                strike_price,           # execution_price: aus contracts.strike_price
                market_price_base,
                market_price_option,    # Ursprünglicher Optionspreis aus contract_aggregates
                imp_vol_percent,
                imp_vol_dec,
                risk_free_rate
            )
            valid_rows.append(data_tuple)

            if len(valid_rows) >= self.batch_size:
                prefiltered_repo.bulk_insert_prefiltered(valid_rows, self.index)
                valid_rows = []

        if valid_rows:
            prefiltered_repo.bulk_insert_prefiltered(valid_rows, self.index)

        raw_conn.close()

    # Die folgenden Methoden bleiben unverändert, werden aber zur Berechnung des risk_free_rate benötigt:
    def get_treasury_rate(self, cursor, date, series_id):
        cursor.execute("SELECT interest_rate FROM treasury_bill WHERE date = ? AND series_id = ?", (date, series_id))
        result = cursor.fetchone()
        if result and result[0] is not None:
            return round(result[0] / 100, 4)
        # Falls keine Daten vorhanden, Suche in beiden Richtungen bis zu 10 Tage
        max_days_back = 10
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
        for offset in range(1, max_days_back + 1):
            prev_date = (date_obj - timedelta(days=offset)).strftime("%Y-%m-%d")
            next_date = (date_obj + timedelta(days=offset)).strftime("%Y-%m-%d")
            cursor.execute("SELECT interest_rate FROM treasury_bill WHERE date = ? AND series_id = ?",
                           (prev_date, series_id))
            result = cursor.fetchone()
            if result and result[0] is not None:
                return round(result[0] / 100, 4)
            cursor.execute("SELECT interest_rate FROM treasury_bill WHERE date = ? AND series_id = ?",
                           (next_date, series_id))
            result = cursor.fetchone()
            if result and result[0] is not None:
                return round(result[0] / 100, 4)
        print(f"⚠ Keine Zinsdaten für {date}, setze 0.0.")
        return 0.0

    def log_linear_interest_interpolation(self, rates, days):
        import numpy as np
        import scipy.interpolate
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
            return ['DTB4WK']
        if remaining_time > 364:
            return ['DTB1YR']
        if remaining_time > 28 and remaining_time <= 84:
            return ['DTB4WK', 'DTB3']
        if remaining_time > 84 and remaining_time <= 182:
            return ['DTB3', 'DTB6']
        if remaining_time > 182 and remaining_time <= 364:
            return ['DTB6', 'DTB1YR']
