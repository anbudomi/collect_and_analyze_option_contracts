import re
import math
import sqlite3
import datetime
from datetime import timedelta


class DataPreparer:
    def __init__(self, raw_database_path, prefiltered_db_path, prepared_db_path, index, index_ticker):
        self.raw_db_path = f"{raw_database_path}/rawdata_{index.lower()}_db.sqlite"
        self.prefiltered_db_path = prefiltered_db_path  # Pfad zur prefiltered_DB
        self.prepared_db_path = prepared_db_path
        self.index = index
        self.index_ticker = index_ticker
        self.batch_size = 100000

    def extract_base_ticker(self, ticker):
        """
        Erwartet ein Format wie: "O:NDX180105C06475000"
        Extrahiert die Bestandteile und gibt einen Basis-String zurück, der ohne den Options-Typ (C/P) ist.
        Beispiel:
          Input:  "O:NDX180105C06475000"
          Output: "O:NDX18010506475000"
        """
        pattern = r"^(O:[A-Z]+)(\d{6})([CP])(\d+)$"
        m = re.match(pattern, ticker)
        if m:
            base = m.group(1) + m.group(2) + m.group(4)
            return base
        else:
            # Falls das Format nicht passt, gebe den Original-Ticker zurück (um Fehler zu vermeiden)
            return ticker

    def process_prepared_data(self, prepared_data_repo):
        """
        Liest alle Call-Einträge aus der prefiltered_DB (für den aktuellen Index)
        und sucht jeweils nach einem passenden Put-Pendant, basierend auf:
          - underlying_ticker
          - execution_price
          - remaining_days
          - und dem extrahierten Base-Ticker (Ticker ohne den Options-Typ)

        Für jedes gefundene Paar wird mittels Call-Put-Parity der theoretische Callpreis
        berechnet und dessen Abweichung (absolute und relative Fehler) ermittelt.
        Liegt die relative Abweichung innerhalb von ±40%, werden **beide** Einträge
        (Call und Put) in die prepared_data_DB übernommen.

        Es werden zudem bis zu 10 gefundene Paare (Details) zur Kontrolle ausgegeben.
        """
        conn = sqlite3.connect(self.prefiltered_db_path)
        cursor = conn.cursor()
        table_name = f"prefiltered_{self.index.lower()}_data"

        # Hole alle Put-Einträge einmalig aus der Datenbank
        put_query_all = f"SELECT * FROM {table_name} WHERE contract_type = 'put'"
        put_rows = cursor.execute(put_query_all).fetchall()

        # Baue ein Dictionary für schnellen Zugriff.
        # Schlüssel: (underlying_ticker, execution_price, remaining_days, base_ticker)
        put_dict = {}
        for row in put_rows:
            put_ticker = row[1]
            underlying_ticker = row[2]
            execution_price = row[8]
            remaining_days = row[6]
            base_ticker = self.extract_base_ticker(put_ticker)
            key = (underlying_ticker, execution_price, remaining_days, base_ticker)
            # Falls mehrere Einträge vorhanden sind, wird hier nur der erste gespeichert
            if key not in put_dict:
                put_dict[key] = row

        # Hole alle Call-Einträge
        query_calls = f"SELECT * FROM {table_name} WHERE contract_type = 'call'"
        call_rows = cursor.execute(query_calls).fetchall()

        accepted_data = []
        printed_pairs_count = 0

        for call_row in call_rows:
            # Annahme: Spaltenreihenfolge entspricht der CREATE TABLE-Reihenfolge:
            # 0: id, 1: ticker, 2: underlying_ticker, 3: date, 4: contract_type,
            # 5: expiration_date, 6: remaining_days, 7: remaining_time, 8: execution_price,
            # 9: market_price_base, 10: market_price_option, 11: implied_vola_percent,
            # 12: implied_vola_dec, 13: risk_free_rate, 14: dividend_yield, 15: BSM,
            # 16: absolute_error, 17: relative_error, 18: moneyness
            call_ticker = call_row[1]
            underlying_ticker = call_row[2]
            date_str = call_row[3]
            expiration_date = call_row[5]
            remaining_days = call_row[6]
            remaining_time = call_row[7]
            execution_price = call_row[8]
            market_price_base = call_row[9]
            call_market_price_option = call_row[10]
            implied_vola_percent = call_row[11]
            implied_vola_dec = call_row[12]
            risk_free_rate = call_row[13]
            dividend_yield = call_row[14]
            BSM_value = call_row[15]
            original_abs_err_call = call_row[16]
            original_rel_err_call = call_row[17]
            moneyness = call_row[18]

            # Extrahiere den Base-Ticker für den Call
            call_base_ticker = self.extract_base_ticker(call_ticker)

            # Schlüssel für das Dictionary
            key = (underlying_ticker, execution_price, remaining_days, call_base_ticker)
            if key not in put_dict:
                continue  # Kein passendes Put-Pendant gefunden

            put_row = put_dict[key]
            put_ticker = put_row[1]
            put_market_price_option = put_row[10]
            original_abs_err_put = put_row[16]
            original_rel_err_put = put_row[17]

            # Berechne den theoretischen Callpreis (Call-Put-Parity)
            call_theoretical = put_market_price_option + (
                    market_price_base - execution_price * math.exp(-risk_free_rate * remaining_time)
            )
            # Berechne die Abweichungen (nur für Filterung und Debug-Ausgabe)
            computed_abs_err = abs(call_theoretical - call_market_price_option)
            computed_rel_err = computed_abs_err / call_market_price_option if call_market_price_option != 0 else float(
                'inf')

            # Filter: Nur wenn die relative Abweichung ≤ 40% ist, wird das Paar übernommen
            if computed_rel_err > 0.4:
                continue

            # Erstelle die Datensätze für die prepared_DB – sowohl Call als auch Put
            prepared_tuple_call = (
                call_ticker,  # ticker
                date_str,  # date
                "call",  # option_type
                execution_price,  # execution_price
                market_price_base,  # market_base_price
                remaining_days,  # remaining_days
                remaining_time,  # remaining_time
                risk_free_rate,  # risk_free_rate
                call_market_price_option,  # market_price_option
                implied_vola_percent,  # implied_vola_percent
                implied_vola_dec,  # implied_vola_dec
                dividend_yield,  # dividend_yield
                BSM_value,  # BSM
                original_abs_err_call,  # absolute_error (aus prefiltered)
                original_rel_err_call,  # relative_error (aus prefiltered)
                moneyness  # moneyness
            )
            prepared_tuple_put = (
                put_ticker,  # ticker
                date_str,  # date
                "put",  # option_type
                execution_price,  # execution_price
                market_price_base,  # market_base_price
                remaining_days,  # remaining_days
                remaining_time,  # remaining_time
                risk_free_rate,  # risk_free_rate
                put_market_price_option,  # market_price_option
                put_row[11],  # implied_vola_percent (aus put_row)
                put_row[12],  # implied_vola_dec (aus put_row)
                put_row[14],  # dividend_yield (aus put_row)
                put_row[15],  # BSM (aus put_row)
                original_abs_err_put,  # absolute_error (aus prefiltered)
                original_rel_err_put,  # relative_error (aus prefiltered)
                put_row[18]  # moneyness (aus put_row)
            )

            accepted_data.append(prepared_tuple_call)
            accepted_data.append(prepared_tuple_put)

            # Batch-Inserts, basierend auf self.batch_size
            if len(accepted_data) >= self.batch_size:
                prepared_data_repo.bulk_insert(accepted_data, self.index)
                accepted_data = []

        conn.close()

        if accepted_data:
            prepared_data_repo.bulk_insert(accepted_data, self.index)
            print(
                f"✅ {((len(accepted_data)) // 2) + printed_pairs_count} Paare (insgesamt {len(accepted_data)} Einträge) in prepared_data_db eingefügt.")
        else:
            print("✅ Alle Paare wurden bereits batchweise eingefügt.")

    def process_prefiltered_data(self, prefiltered_repo):
        """
        Liest die Rohdaten, führt den Join zwischen contract_aggregates und contracts
        sowie weitere Abfragen durch und fügt die Daten in die prefiltered_DB ein.
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

        # Lade Indexdaten: Mapping: date -> close (Preisindex)
        raw_cursor.execute("SELECT date, close FROM index_data")
        index_data = {row[0]: row[1] for row in raw_cursor.fetchall()}

        # Lade Performance-Index-Daten: Mapping: date -> close
        raw_cursor.execute("SELECT date, close FROM performance_index")
        perf_index_data = {row[0]: row[1] for row in raw_cursor.fetchall()}

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

            # Risk-free rate (über Interpolation)
            risk_free_rate = self.compute_risk_free_rate(raw_cursor, date_str, remaining_days)

            # Dividend Yield berechnen (unter Verwendung der Performance- und Preisindexdaten)
            div_yield = self.get_dividend_yield(perf_index_data, index_data, date_str, trade_date)

            # BSM Preis berechnen, abhängig vom contract_type (Call/Put)
            bsm = self.get_bsm_price(contract_type, market_price_base, strike_price, risk_free_rate, div_yield,
                                     imp_vol_dec, remaining_time)

            # Absolute und relative Fehler berechnen:
            abs_err = round(abs(market_price_option - bsm), 2)
            rel_err = round(abs_err / market_price_option, 4) if market_price_option != 0 else 0.0

            # Moneyness
            if contract_type == "call":
                moneyness = round((market_price_base - strike_price) / strike_price, 4)
            elif contract_type == "put":
                moneyness = round((strike_price - market_price_base) / strike_price, 4)

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
                risk_free_rate,
                div_yield,
                bsm,
                abs_err,
                rel_err,
                moneyness
            )
            valid_rows.append(data_tuple)

            if len(valid_rows) >= self.batch_size:
                prefiltered_repo.bulk_insert_prefiltered(valid_rows, self.index)
                valid_rows = []

        if valid_rows:
            prefiltered_repo.bulk_insert_prefiltered(valid_rows, self.index)

        raw_conn.close()

    def get_bsm_price(self, contract_type, S, K, r, q, sigma, T):
        """
        Berechnet den Black-Scholes-Merton Preis für einen Call oder Put.

        Parameter:
          contract_type: 'call' oder 'put'
          S: Underlying-Preis (market_price_base)
          K: Strike-Price (aus contracts.strike_price)
          r: Risikofreier Zinssatz (risk_free_rate)
          q: Dividendenrendite (dividend_yield)
          sigma: Implizite Volatilität (in Dezimal, z. B. 0.20 für 20%)
          T: Verbleibende Zeit in Jahren (z. B. 0.25)

        Rückgabe:
          Der BSM-Preis, gerundet auf 2 Nachkommastellen.
        """
        # Sonderfälle: Falls T oder sigma 0 oder negativ sind, liefert der innere Wert den
        # intrinsischen Wert.
        if T <= 0 or sigma <= 0:
            if contract_type.lower() == "call":
                return max(0, S - K)
            elif contract_type.lower() == "put":
                return max(0, K - S)
            else:
                return 0.0

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        # Standard-Normalverteilungsfunktion
        N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))

        if contract_type.lower() == "call":
            price = S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)
        elif contract_type.lower() == "put":
            price = K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)
        else:
            # Fallback: Standardmäßig Call-Berechnung
            print("Weder call noch put gefunden")
            price = -100.0          # Um Fehler festzustellen, unrealistischen Wert einfügen

        return round(price, 2)

    # --- Hilfsmethoden für Dividend Yield ------------------

    def get_nearest_value(self, data_dict, target_date_str, max_offset=10):
        """
        Sucht in data_dict (Mapping date -> Wert) nach target_date_str.
        Falls das Datum existiert, wird auch überprüft, ob der zugehörige Wert ungleich 0 ist.
        Falls nicht, wird in 1-Tages-Schritten (zunächst rückwärts, dann vorwärts)
        nach einem Datum gesucht, dessen Wert nicht 0 ist.
        Gibt None zurück, falls innerhalb von max_offset Tagen kein gültiger Wert gefunden wird.
        """
        if target_date_str in data_dict and data_dict[target_date_str] != 0:
            return data_dict[target_date_str]

        target_date = datetime.datetime.strptime(target_date_str, "%Y-%m-%d")

        for offset in range(1, max_offset + 1):
            prev_date = (target_date - timedelta(days=offset)).strftime("%Y-%m-%d")
            if prev_date in data_dict and data_dict[prev_date] != 0:
                return data_dict[prev_date]

            next_date = (target_date + timedelta(days=offset)).strftime("%Y-%m-%d")
            if next_date in data_dict and data_dict[next_date] != 0:
                return data_dict[next_date]

        return None

    def get_dividend_yield(self, perf_index_data, price_index_data, current_date_str, trade_date):
        """
        Berechnet die Dividendenrendite:
        div_yield = ( (Perf_current / Perf_prev) / (Price_current / Price_prev) ) - 1
        Falls Daten für das Ziel- oder das Vorjahrsdatum nicht gefunden werden, wird 0.0 zurückgegeben.
        """
        prev_year_date_str = (trade_date - timedelta(days=365)).strftime("%Y-%m-%d")
        perf_current = self.get_nearest_value(perf_index_data, current_date_str)
        perf_prev = self.get_nearest_value(perf_index_data, prev_year_date_str)
        price_current = self.get_nearest_value(price_index_data, current_date_str)
        price_prev = self.get_nearest_value(price_index_data, prev_year_date_str)
        #print(perf_current, perf_prev, price_current, price_prev)
        if None in (perf_current, perf_prev, price_current, price_prev):
            return 0.0
        ratio = (perf_current / perf_prev) / (price_current / price_prev)
        div_yield = ratio - 1
        return round(div_yield, 4)

    # --- Die übrigen Methoden (risk-free rate etc.) bleiben unverändert ---
    def compute_risk_free_rate(self, cursor, date_str, remaining_days):
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

    def get_treasury_rate(self, cursor, date, series_id):
        cursor.execute("SELECT interest_rate FROM treasury_bill WHERE date = ? AND series_id = ?", (date, series_id))
        result = cursor.fetchone()
        if result and result[0] is not None:
            return round(result[0] / 100, 4)
        max_days_back = 10
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
        for offset in range(1, max_days_back + 1):
            prev_date = (date_obj - timedelta(days=offset)).strftime("%Y-%m-%d")
            next_date = (date_obj + timedelta(days=offset)).strftime("%Y-%m-%d")
            cursor.execute("SELECT interest_rate FROM treasury_bill WHERE date = ? AND series_id = ?", (prev_date, series_id))
            result = cursor.fetchone()
            if result and result[0] is not None:
                return round(result[0] / 100, 4)
            cursor.execute("SELECT interest_rate FROM treasury_bill WHERE date = ? AND series_id = ?", (next_date, series_id))
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
