import requests
import time
from concurrent.futures import ThreadPoolExecutor
from fredapi import Fred
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from queue import Queue, Empty
from threading import Thread
import asyncio
import aiohttp
import datetime
import logging
import pandas as pd

db_queue = Queue()

class PolygonApiClient:

    #region Client-Init : Initiiert PolygonClient
    def __init__(
            self,
            api_key,
            us_holidays,
            start_date,
            end_date,
            limit,
            contract_type,
            underlying_ticker,
            expired,
            db_repository,
            closing_candles_start_date,
            closing_candles_end_date,
            interest_rate_start_date,
            interest_rate_end_date,
            implied_volatility_start_date,
            implied_volatility_end_date,
            index_data_ticker,
            implied_volatility_ticker,
            fred_api_key,
            batch_size = 20000,
            max_workers = 50,
    ):
        self.base_url = "https://api.polygon.io"
        self.api_key = api_key
        self.fred_api_key = fred_api_key
        self.us_holidays = us_holidays
        self.start_date = start_date
        self.end_date = end_date
        self.limit = limit
        self.contract_type = contract_type
        self.underlying_ticker = underlying_ticker
        self.expired = expired
        self.db_repository = db_repository
        self.closing_candles_start_date = closing_candles_start_date
        self.closing_candles_end_date = closing_candles_end_date
        self.interest_rate_start_date = interest_rate_start_date
        self.interest_rate_end_date = interest_rate_end_date
        self.implied_volatility_start_date = implied_volatility_start_date
        self.implied_volatility_end_date = implied_volatility_end_date
        self.implied_volatility_ticker = implied_volatility_ticker
        self.index_data_ticker = index_data_ticker
        self.batch_size = batch_size
        self.max_workers = max_workers

        # Initialize a persistent session with default headers
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'PolygonApiClient/1.0'})

        # Precompute clamp date
        self.clamp_date_str = "2022-12-31"
        self.clamp_date_obj = datetime.datetime.strptime(self.clamp_date_str, "%Y-%m-%d")

        # Initialize logger
        self.logger = logging.getLogger(__name__)
    #endregion

    # ----------------------------------------------------------------
    # 1) Helfer-Funktionen
    # ----------------------------------------------------------------

    #region Helfer-Funktionen : Prüft verschiedene Rahmenbedingungen
    def ensure_datetime(self, date):
        """Konvertiert einen String in ein `datetime`-Objekt, falls nötig."""
        if isinstance(date, datetime.datetime):
            return date
        elif isinstance(date, str):
            return datetime.datetime.strptime(date, "%Y-%m-%d")
        else:
            raise TypeError("Invalid date format. Expected datetime or string.")

    def get_trading_days(self):
        return [
            date for date in (self.start_date + datetime.timedelta(days=i)
                              for i in range((self.end_date - self.start_date).days + 1))
            if self.is_trading_day(date)
        ]

    def is_trading_day(self, date: datetime.datetime) -> bool:
        """
        Prüft, ob ein Datum ein US-Handelstag ist (kein Wochenende oder US-Feiertag).
        """
        if not isinstance(date, datetime.datetime):
            raise TypeError(f"'date' must be a datetime, got {type(date).__name__}.")

        # Weekend check (Monday=0, Sunday=6)
        if date.weekday() >= 5:
            return False

        # U.S. holiday check
        date_str = date.strftime("%Y-%m-%d")
        if date_str in self.us_holidays:
            return False

        return True

    def validate_date_range(self):
        """Überprüft, dass: start_date <= end_date."""
        if self.start_date > self.end_date:
            raise ValueError("Start date cannot be after end date.")
        return True
    #endregion

    # ----------------------------------------------------------------
    # 2) Error-Handling
    # ----------------------------------------------------------------

    #region Error-Handling
    def is_transient_error(exception):
        """ Prüft, ob es sich um einen transienten Fehler handelt (429, 5XX) oder eine allgemeine requests-Exception. """
        if isinstance(exception, requests.exceptions.RequestException):
            response = getattr(exception, 'response', None)
            if response and response.status_code in [429] + list(range(500, 600)):
                return True
            return True  # Alle requests-Exceptions sollen retryen
        return False

    @retry(
        stop=stop_after_attempt(5),  # Max. 5 Versuche
        wait=wait_exponential(multiplier=2, min=1, max=30),  # Exponentielles Warten (1s, 2s, 4s, 8s...)
        retry=retry_if_exception(is_transient_error),  # Nur bei transienten Fehlern erneut versuchen
    )
    #endregion

    # ----------------------------------------------------------------
    # 3) Datensammlung von Contracts-Daten
    # ----------------------------------------------------------------

    #region Datensammlung für Contracts : Sammelt alle Contracts für die angegebenen Parameter
    def run_fetch_contracts(self, max_workers=10):
        """ Startet mehrere Worker-Threads zur Parallelverarbeitung von Trading-Days. """
        trading_days = self.get_trading_days()
        q = Queue()

        # 🔹 Füge alle Trading-Days in die Queue
        for day in trading_days:
            q.put(day)

        # 🔹 Starte den DB-Worker
        db_thread = Thread(target=self.db_worker, daemon=True)
        db_thread.start()

        # 🔹 Starte die Worker-Threads
        workers = []
        for _ in range(max_workers):
            t = Thread(target=self.worker, args=(q,))
            t.start()
            workers.append(t)

        # 🔹 Warte, bis alle Tasks erledigt sind
        q.join()

        for t in workers:
            t.join()

        # 🔹 DB-Worker beenden
        db_queue.put(None)
        db_thread.join()

    def worker(self, q):
        """Verarbeitet Trading-Days aus der Queue und ruft fetch_and_store_contracts() auf."""
        while True:
            try:
                day = q.get(block=True, timeout=5)  # Blockiert bis zu 5 Sekunden
            except Empty:
                break  # Beende Worker, falls die Queue leer bleibt

            try:
                self.fetch_and_store_contracts(day)
            except Exception as e:
                self.logger.error(f"Error on {day}: {e}")

            q.task_done()

    def db_worker(self):
        """ Arbeitet alle DB-Inserts aus der Queue ab. """
        while True:
            chunk = db_queue.get()
            if chunk is None:
                break  # Beenden-Signal erhalten
            try:
                self.db_repository.insert_contracts_bulk(chunk)
            except Exception as e:
                with open("failed_inserts.log", "a") as log_file:
                    log_file.write(f"DB Insert Error: {str(e)}\n")
            db_queue.task_done()

    def fetch_and_store_contracts(self, current_date):
        try:
            if isinstance(current_date, datetime.datetime):
                current_date = current_date.date()

            self.logger.info(f"📅 Verarbeite Trading-Day: {current_date.strftime('%Y-%m-%d')}")

            all_options = []
            next_url = (
                f"{self.base_url}/v3/reference/options/contracts"
                f"?underlying_ticker={self.underlying_ticker}"
                f"&as_of={current_date.strftime('%Y-%m-%d')}"
                f"&expired={self.expired}"
                f"&contract_type={self.contract_type}"
                f"&limit={self.limit}"
            )

            while next_url:
                if "apiKey" not in next_url:
                    connector = "&" if "?" in next_url else "?"
                    next_url = f"{next_url}{connector}apiKey={self.api_key}"

                data = self.fetch_with_retry(next_url)

                # 🛑 Falls kein gültiges JSON-Daten-Objekt zurückgegeben wird, sofort abbrechen!
                if not isinstance(data, dict):
                    self.logger.error(f"❌ API-Fehler: Ungültige Antwort für {current_date.strftime('%Y-%m-%d')}")
                    return

                results = data.get("results", [])
                all_options.extend(results)

                # 🛑 Falls `next_url` nicht existiert, breche die Schleife ab.
                next_url = data.get("next_url", None)

            if not all_options:
                self.logger.warning(f"⚠ Keine Contracts für {current_date.strftime('%Y-%m-%d')} gefunden.")
                return

            # 🔹 chunk_size-Fix für Performance
            chunk_size = self.batch_size
            rows_for_db = [
                {
                    "ticker": r["ticker"],
                    "underlying_ticker": r["underlying_ticker"],
                    "cfi": r["cfi"],
                    "contract_type": r["contract_type"],
                    "exercise_style": r["exercise_style"],
                    "expiration_date": self.clamp_date_str if datetime.datetime.strptime(r["expiration_date"],
                                                                                         "%Y-%m-%d") > self.clamp_date_obj else
                    r["expiration_date"],
                    "primary_exchange": r["primary_exchange"],
                    "shares_per_contract": r["shares_per_contract"],
                    "strike_price": r["strike_price"],
                    "date": current_date.strftime("%Y-%m-%d")
                }
                for r in all_options
            ]

            for i in range(0, len(rows_for_db), chunk_size):
                chunk = rows_for_db[i:i + chunk_size]
                db_queue.put(chunk)
                self.logger.info(
                    f"🔄 Starte Bulk-Insert für {len(chunk)} Contracts am {current_date.strftime('%Y-%m-%d')}...")

            self.logger.info(f"✅ Verarbeitung abgeschlossen für {current_date.strftime('%Y-%m-%d')}")

        except Exception as e:
            self.logger.error(f"❌ Fehler für {current_date.strftime('%Y-%m-%d')} - {str(e)}")

    #endregion

    # ----------------------------------------------------------------
    # 4) Datensammlung von Aggregates
    # ----------------------------------------------------------------

    # region Datensammlung für Aggregates : Sammelt alle Aggregates für die angegebenen Parameter
    def fetch_and_store_aggregates(self, max_workers=10, batch_size=20000):
        """
        Holt historische Aggregates-Daten für alle Contracts aus der Datenbank und speichert sie.
        """
        query = """
            SELECT ticker, MIN(date) AS first_date, MAX(expiration_date) AS last_date
            FROM contracts
            GROUP BY ticker
            ORDER BY first_date ASC
        """
        contracts = self.db_repository.connection.cursor().execute(query).fetchall()
        total_contracts = len(contracts)

        if not contracts:
            self.logger.error(f"⚠️ Keine Contracts gefunden! Abbruch der Aggregates-Sammlung.")
            return

        self.logger.info(f"🔍 {total_contracts} Contracts gefunden. Starte Fetch...")

        def fetch_aggregate_data(contract):
            ticker, start_date, expiration_date = contract
            start_date, expiration_date = map(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d"),
                                              [start_date, expiration_date])

            if self.db_repository.aggregate_exists(ticker, start_date.strftime('%Y-%m-%d'),
                                                   expiration_date.strftime('%Y-%m-%d')):
                return None

            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{expiration_date.strftime('%Y-%m-%d')}?adjusted=true&apiKey={self.api_key}"
            result = self.fetch_aggregate_sync(url, ticker)

            if result and not result.get("error") and result.get("data"):
                return result["data"]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            all_aggregates = list(filter(None, executor.map(fetch_aggregate_data, contracts)))

        if all_aggregates:
            self.db_repository.insert_contract_aggregates_bulk(sum(all_aggregates, []))

        self.logger.info(
            f"🎉 Fetch & Store Aggregates abgeschlossen! 🏁 Total Contracts: {total_contracts}, Total Aggregates gespeichert!")

    def fetch_aggregate_sync(self, url, ticker):
        """
        Holt Aggregates-Daten für einen einzelnen Contract aus der API.
        """
        try:
            data = self.fetch_with_retry(url)

            if not data or data.get("status") != "OK":
                error_message = data.get("message", "Unbekannter API-Fehler")
                self.logger.error(f"🚨 API-Fehler für {ticker}: {error_message}")
                return {"ticker": ticker, "data": None, "error": True, "url": url}

            results = data.get("results", [])
            return {
                "ticker": ticker,
                "data": [
                    {
                        "contract_ticker": ticker,
                        "c": agg.get("c"),
                        "h": agg.get("h", 0.0),
                        "l": agg.get("l", 0.0),
                        "o": agg.get("o", 0.0),
                        "t": agg.get("t"),
                        "v": agg.get("v", 0),
                        "vw": agg.get("vw", 0.0),
                        "date": datetime.datetime.utcfromtimestamp(agg["t"] / 1000).strftime("%Y-%m-%d")
                    }
                    for agg in results if agg.get("c") is not None and agg.get("t") is not None
                ],
                "error": False,
                "url": url
            }

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Request-Fehler für {ticker}: {e}")
            return {"ticker": ticker, "data": None, "error": True, "url": url, "exception": str(e)}

        except Exception as e:
            self.logger.error(f"❌ Fehler beim Abrufen von Aggregates für {ticker}: {e}")
            return {"ticker": ticker, "data": None, "error": True, "url": url, "exception": str(e)}

    async def fetch_aggregate_async(self, session, ticker, start_date, end_date):
        """
        Holt Aggregates für einen einzelnen Ticker asynchron mit `aiohttp`.
        """
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&apiKey={self.api_key}"

        try:
            async with session.get(url) as response:
                data = await response.json()

                # Falls API antwortet, aber kein Data vorhanden ist
                if data.get("resultsCount", 0) == 0:
                    return None

                results = data.get("results", [])
                valid_results = [
                    {
                        "contract_ticker": ticker,
                        "c": agg.get("c"),
                        "h": agg.get("h", 0.0),
                        "l": agg.get("l", 0.0),
                        "n": agg.get("n", 0),
                        "o": agg.get("o", 0.0),
                        "t": agg.get("t"),
                        "v": agg.get("v", 0),
                        "vw": agg.get("vw", 0.0),
                        "date": datetime.datetime.utcfromtimestamp(agg["t"] / 1000).strftime("%Y-%m-%d")
                    }
                    for agg in results if agg.get("c") is not None and agg.get("t") is not None
                ]
                return valid_results

        except Exception as e:
            self.logger.error(f"❌ API Fehler für {ticker}: {e} | URL: {url}")
            return None

    async def fetch_and_store_aggregates_async(self):
        """
        Holt & speichert alle Aggregates asynchron mit `aiohttp`.
        """
        query = """
            SELECT ticker, MIN(date) AS first_date, MAX(expiration_date) AS last_date
            FROM contracts
            GROUP BY ticker
            ORDER BY first_date ASC
        """
        c = self.db_repository.connection.cursor()
        c.execute(query)
        contracts = c.fetchall()
        total_contracts = len(contracts)

        if not contracts:
            self.logger.error("⚠️ Keine Contracts gefunden!")
            return

        total_aggregates = 0  # Gesamtanzahl
        processed_contracts = 0  # Anzahl der verarbeiteten Contracts
        aggregate_buffer = []  # **Buffer für DB-Writes**

        self.logger.info(f"🔍 {total_contracts} Contracts gefunden. Starte Fetch...")

        async with aiohttp.ClientSession() as session:
            for i in range(0, len(contracts), self.max_workers):
                batch = contracts[i:i + self.max_workers]  # Nimm `max_workers` Contracts für gleichzeitige Anfragen

                # ❗ Hier sicherstellen, dass `fetch_aggregate_async` awaited wird
                results = await asyncio.gather(
                    *(self.fetch_aggregate_async(session, ticker, start_date, expiration_date) for
                      ticker, start_date, expiration_date in batch)
                )

                processed_contracts += len(batch)

                for result in results:
                    if result:
                        aggregate_buffer.extend(result)
                        total_aggregates += len(result)

                # **Schreibe Buffer in die DB, wenn voll**
                if len(aggregate_buffer) >= self.batch_size:
                    self.db_repository.insert_contract_aggregates_bulk(aggregate_buffer)
                    aggregate_buffer = []  # **Buffer leeren**
                    self.logger.info(f"💾 {total_aggregates} Aggregates gespeichert!")

                # **Fortschritt nur alle 1000 Contracts ausgeben**
                if processed_contracts % 1000 == 0:
                    self.logger.info(
                        f"📊 {processed_contracts}/{total_contracts} Contracts verarbeitet... | Gesamt Aggregates: {total_aggregates}")

            # **Letzten Buffer speichern**
            if aggregate_buffer:
                self.db_repository.insert_contract_aggregates_bulk(aggregate_buffer)
                self.logger.info(f"💾 {total_aggregates} Aggregates gespeichert!")

        self.logger.info(f"🎉 Fetch & Store abgeschlossen! 🏁 Total Aggregates: {total_aggregates}")

    def run_fetch_and_store_aggregates(self):
        """
        Startet die asynchrone Verarbeitung im Event-Loop.
        Falls bereits ein Event-Loop läuft (z.B. in Jupyter Notebooks oder modernen Python-Versionen), wird `asyncio.run()` verwendet.
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.ensure_future(self.fetch_and_store_aggregates_async())
            loop.run_until_complete(future)
        except RuntimeError:
            asyncio.run(
                self.fetch_and_store_aggregates_async())  # ✅ Sicherstellen, dass ein neuer Event-Loop gestartet wird

    #endregion

    # ----------------------------------------------------------------
    # 5) Zusätzliche Daten : Close-Werte, FRED-Daten, implizite Volatilität
    # ----------------------------------------------------------------

    #region Datensammlung von Yfinance-Daten : Close-Werte des angegebenen Index und implizite Volatilität

    def fetch_yfinance_data(self, ticker, start_date, end_date, insert_func):
        """
        Holt Daten von Yahoo Finance und speichert sie mit einer definierten Insert-Funktion.
        """
        max_retries = 5
        wait_time = 10  # Start-Wartezeit in Sekunden

        for attempt in range(max_retries):
            try:
                self.logger.info(f"📊 Fetching data for {ticker}... (Try {attempt + 1}/{max_retries})")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                # ✅ Sicherstellen, dass `data` ein DataFrame ist
                if not isinstance(data, pd.DataFrame):
                    self.logger.error(f"⚠️ Fehler: `yf.download()` hat unerwartete Daten zurückgegeben. Inhalt:")
                    raise ValueError("yfinance returned a non-DataFrame object")

                # ✅ Sicherstellen, dass `data` Spalten hat
                if data.empty:
                    self.logger.warning(f"⚠️ Keine Daten für {ticker} erhalten. Erneuter Versuch...")
                    raise ValueError("Empty Data")

                # ✅ Daten formatieren
                formatted_data = self.format_yfinance_data(data)

                if not isinstance(formatted_data, pd.DataFrame):
                    self.logger.error(f"❌ Fehler: `format_yfinance_data()` hat kein DataFrame zurückgegeben!")
                    raise ValueError("format_yfinance_data returned a non-DataFrame object")

                if formatted_data.empty:
                    self.logger.error(f"⚠️ Keine formatierbaren Daten für {ticker} erhalten. Erneuter Versuch...")
                    raise ValueError("Formatted Data Empty")

                # ✅ Index zurücksetzen
                formatted_data.reset_index(inplace=True)

                # ✅ Daten in die DB einfügen
                insert_func(formatted_data, ticker)
                self.logger.info(f"✅ Daten gespeichert für {ticker}")
                return

            except Exception as e:
                if "Too Many Requests" in str(e) or "Empty Data" in str(e):
                    self.logger.error(f"⏳ Warte {wait_time} Sekunden wegen Rate-Limit oder leeren Daten...")
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponentielles Warten
                else:
                    self.logger.error(f"❌ Fehler für {ticker}: {e}")
                    break  # Kein erneuter Versuch bei anderen Fehlern

    #Der alte Stand hat ohne diese Funktion funktioniert, allerdings hat YahooFinance ihre API verändert
    def format_yfinance_data(self, input_df):
        # ✅ Falls MultiIndex vorhanden ist, entferne die erste Ebene ("Price")
        if isinstance(input_df.columns, pd.MultiIndex):
            input_df.columns = input_df.columns.droplevel(1)  # Entferne die "Ticker"-Ebene

        # ✅ Index zurücksetzen, damit das Datum eine Spalte wird
        input_df = input_df.reset_index()

        # ✅ Spaltennamen normalisieren (yfinance verwendet manchmal "Date" statt "Datetime")
        if "Date" in input_df.columns:
            input_df.rename(columns={"Date": "Datetime"}, inplace=True)

        # ✅ Nur relevante Spalten behalten
        required_columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
        available_columns = [col for col in required_columns if col in input_df.columns]

        return input_df[available_columns]  # Rückgabe nur der relevanten Daten

    #endregion

    #region Datensammlung von FRED-Daten : Sammelt Zins-Daten
    def fetch_fred_data(self, series_id):
        """
        Holt FRED-Daten für die angegebene Serie und speichert sie in der Datenbank.
        """
        try:
            fred = Fred(api_key=self.fred_api_key)

            # API-Verfügbarkeit prüfen
            if not fred:
                self.logger.error(f"❌ FRED API nicht erreichbar!")
                return

            data = fred.get_series(series_id, self.interest_rate_start_date, self.interest_rate_end_date)

            if data is None or (hasattr(data, "empty") and data.empty):
                self.logger.warning(f"⚠️ Keine Daten für FRED-Serie {series_id} erhalten.")
                return

            self.db_repository.insert_data_interest_rates(data, series_id)
            self.logger.info(f"✅ FRED-Daten ({series_id}) erfolgreich gespeichert.")

        except Exception as e:
            self.logger.error(f"❌ Fehler beim Abruf von FRED-Daten ({series_id}): {e}")
    #endregion

    # ----------------------------------------------------------------
    # 6) Helferfunktionen für API-Anfragen
    # ----------------------------------------------------------------

    #region Helferfunktionen für API-Handling
    def fetch_with_retry(self, url, max_retries=5, timeout=10, base_backoff=1, max_backoff=60):
        """
        Führt eine HTTP-Anfrage mit intelligenter Retry-Logik durch.
        """
        backoff = base_backoff  # Startwert für exponentielles Warten

        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, timeout=timeout)

                # 🛑 Wenn der Statuscode in den Fehlerbereich fällt, behandeln wir ihn direkt:
                status_code = resp.status_code

                if status_code == 429:
                    self.logger.warning(
                        f"🚨 API-Rate-Limit (429) erreicht! Warte {backoff} Sekunden... Versuch {attempt + 1}/{max_retries}")
                elif 500 <= status_code < 600:
                    self.logger.warning(
                        f"🔥 Server-Fehler {status_code}. Warte {backoff} Sekunden... Versuch {attempt + 1}/{max_retries}")
                elif status_code >= 400:  # Alle anderen Fehler (kein Retry nötig)
                    self.logger.error(f"❌ Nicht-retrybarer Fehler {status_code}: {resp.text}")
                    return None  # Kein Retry, sofort abbrechen.

                # ✅ JSON-Daten validieren
                try:
                    return resp.json()  # Erfolgreiche Antwort zurückgeben
                except ValueError:
                    self.logger.error(f"❌ Ungültige JSON-Antwort von {url}: {resp.text}")
                    return None  # Fehlerhafte JSON-Daten → Kein Retry, abbrechen.

            except requests.exceptions.ConnectionError:
                self.logger.error(f"❌ Netzwerk-Fehler! Verbindung zum Server fehlgeschlagen.")
            except requests.exceptions.Timeout:
                self.logger.warning(
                    f"⚠ Timeout-Fehler! Warte {backoff} Sekunden... Versuch {attempt + 1}/{max_retries}")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"❌ Unerwarteter Fehler bei Anfrage: {e}")
                break  # Kein Retry, wenn Fehler nicht bekannt ist

            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)  # Exponentielles Wachstum begrenzen.

        self.logger.error(f"❌ Max. Versuche erreicht: {url} konnte nicht geladen werden.")
        return None  # Wenn alle Versuche scheitern

    #endregion