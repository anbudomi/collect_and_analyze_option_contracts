import os
import logging
import dotenv
import datetime
import asyncio
from data_collection.database import DatabaseType
from data_analysis.datahandling import DataHandler, DataAnalyzer, DataPreparer
from data_collection.api import PolygonApiClient
from data_collection.database import get_collection_database_repository
from data_analysis.database import get_analysis_database_repository
import shutil

#.env frühestmöglich auslesen!
dotenv.load_dotenv()

# ----------------------------------------------------------------
# 1) Helfer-Funktionen und Logging
# ----------------------------------------------------------------

#region 1) Helfer-Funktionen und Logging
def parse_boolean(value):
    """Wandelt String in Boolean um."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise ValueError(f"Cannot parse '{value}' as a boolean.")

def setup_logging(log_level=logging.INFO):
    """
    Erstellt eine neue Log-Struktur mit Timestamp und archiviert alte Logs.
    Struktur:
    - error_logging/
      - archiv/               (Alte Logs)
      - YYYYMMDD_HHMM/        (Aktuelle Logs)
        - error_log.txt        (Allgemeine Fehler)
        - api_spx_contracts.txt    (SPX Contracts-Fehler)
        - api_ndx_contracts.txt    (NDX Contracts-Fehler)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_directory = os.path.join(script_dir, "error_logging")
    archive_directory = os.path.join(log_directory, "archiv")

    # Erstelle Verzeichnisse falls sie nicht existieren
    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(archive_directory, exist_ok=True)

    # Aktuellen Log-Ordner erstellen
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    current_log_folder = os.path.join(log_directory, timestamp)
    os.makedirs(current_log_folder, exist_ok=True)

    # Archivierung alter Log-Ordner
    for folder in os.listdir(log_directory):
        folder_path = os.path.join(log_directory, folder)
        if os.path.isdir(folder_path) and folder != "archiv" and folder != timestamp:
            shutil.move(folder_path, os.path.join(archive_directory, folder))

    # Allgemeines Logging
    log_path = os.path.join(current_log_folder, "error_log.txt")

    # Logging Setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Stream Logging (Konsole)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Datei-Logging (Allgemeine Fehler)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"🔹 Logging initialisiert mit neuem Ordner: {current_log_folder}")

    return logger, current_log_folder



# Mapping aus der .env einlesen und in Dicts umwandeln
def parse_env_mapping(env_var):
    """Liest eine Komma-separierte Mapping-Zeichenkette aus der .env und wandelt sie in ein Dict um."""
    mapping = {}
    raw_mapping = os.getenv(env_var, "")
    if raw_mapping:
        for entry in raw_mapping.split(","):
            key, value = entry.split(":")
            mapping[key.strip().upper()] = value.strip()
    return mapping

async def run_aggregates(client):
    await client.fetch_and_store_aggregates_async()

#endregion

def main():
    """Hauptfunktion des Skripts."""

    #initiiert logger
    logger, current_log_folder = setup_logging()

    # ----------------------------------------------------------------
    # 2) Aufruf Datensammlung
    # ----------------------------------------------------------------

    #region 2) Aufruf Datensammlung
    if parse_boolean(os.getenv('RUN_DATA_COLLECTION')):

        # Mappings für Index- und Volatilitätsdaten aus der .env-Datei
        index_data_mapping = parse_env_mapping("INDEX_DATA_MAPPING")
        implied_volatility_mapping = parse_env_mapping("IMPLIED_VOLATILITY_MAPPING")

        # Liste aller Underlying Ticker und Contract Types aus der .env
        underlying_tickers = os.getenv('POLYGON_UNDERLYING_TICKER', 'UNKNOWN').replace(" ", "").upper().split(",")
        contract_types = os.getenv('POLYGON_CONTRACT_TYPE', 'UNKNOWN').replace(" ", "").lower().split(",")

        # Erstelle für jeden Ticker ein eigenes Datenbank-Repository
        db_repositories = get_collection_database_repository(DatabaseType.SQLITE)

        # ----------------------------------------------------------------
        # 2.1) Debugging Helfer-Funktion, um Tabellen zu entfernen
        # ----------------------------------------------------------------

        #region 2.1) Debugging Helfer-Funktion, um Tabellen zu entfernen
        if parse_boolean(os.getenv('RUN_DROP_TABLES', "false")):
            target_dbs = os.getenv('TARGET_DB', "").replace(" ", "").split(",")
            tables_to_drop = os.getenv('TABLES_TO_DROP', "").replace(" ", "").split(",")

            # Extrahiere alle Datenbankpfade aus den db_repositories
            db_paths = [repo.filename for repo in db_repositories.values() if hasattr(repo, 'filename')]

            # Falls eine explizite Liste von Datenbanken in .env angegeben ist, verwende diese stattdessen
            if target_dbs and target_dbs[0]:  # Falls nicht leer
                db_paths = target_dbs

            if db_paths and tables_to_drop:
                print(f"🔴 Lösche Tabellen {tables_to_drop} in den Datenbanken {db_paths}...")
                list(db_repositories.values())[0].drop_tables(db_paths, tables_to_drop)  # Irgendeine Instanz nutzen
            else:
                print("⚠️ `RUN_DROP_TABLES` ist aktiviert, aber `TARGET_DB` oder `TABLES_TO_DROP` ist nicht definiert.")
        #endregion

        # ----------------------------------------------------------------
        # 2.2) Iteration über alle angegebenen Ticker
        # ----------------------------------------------------------------

        #region 2.2) Iteration über alle angegebenen Ticker
        for underlying_ticker in underlying_tickers:
            # Richtige Werte für Index und Volatilität aus der .env holen
            index_ticker = index_data_mapping.get(underlying_ticker, None)
            volatility_ticker = implied_volatility_mapping.get(underlying_ticker, None)

            if index_ticker is None or volatility_ticker is None:
                logger.warning(f"No index or volatility ticker mapping found for {underlying_ticker}. Skipping...")
                continue  # Falls kein Mapping existiert, überspringen

            db_repository = db_repositories[underlying_ticker]  # Die richtige DB für diesen Ticker

            start_date = datetime.datetime.strptime(os.getenv('POLYGON_START_DATE'), "%Y-%m-%d")
            end_date = datetime.datetime.strptime(os.getenv('POLYGON_END_DATE'), "%Y-%m-%d")

            # ✅ **Client wird nur EINMAL pro `underlying_ticker` erstellt!**

            #region Initialisierung Client
            client = PolygonApiClient(
                api_key=os.getenv('POLYGON_API_KEY'),
                limit=os.getenv('POLYGON_LIMIT'),
                start_date=start_date,
                end_date=end_date,
                contract_type=None,  # Wird später dynamisch gesetzt
                underlying_ticker=underlying_ticker,
                expired=parse_boolean(os.getenv('POLYGON_EXPIRED')),
                us_holidays=os.getenv('POLYGON_US_HOLIDAYS', "").split(","),
                db_repository=db_repository,
                closing_candles_start_date=os.getenv('CLOSING_CANDLES_START_DATE'),
                closing_candles_end_date=os.getenv('CLOSING_CANDLES_END_DATE'),
                interest_rate_start_date=os.getenv('INTEREST_RATE_START_DATE'),
                interest_rate_end_date=os.getenv('INTEREST_RATE_END_DATE'),
                implied_volatility_start_date=os.getenv('IMPLIED_VOLATILITY_START_DATE'),
                implied_volatility_end_date=os.getenv('IMPLIED_VOLATILITY_END_DATE'),
                index_data_ticker=index_ticker,  # Dynamisch aus .env
                implied_volatility_ticker=volatility_ticker,  # Dynamisch aus .env
                fred_api_key=os.getenv('FRED_API_KEY'),
                batch_size=int(os.getenv('BATCH_COLLECTION_SIZE')),
            )
            #endregion

            # ----------------------------------------------------------------
            # 2.2.1) Ausführung der einzelnen Skripte
            # ----------------------------------------------------------------

            #region 2.2.1) Ausführung der einzelnen Skripte
            try:
                #region Ausführung der Datenbank-Migration
                if parse_boolean(os.getenv('RUN_MIGRATION')):
                    db_repository.collection_db_migrate()
                    logger.info(f"Database migrated successfully for {underlying_ticker}.")
                #endregion

                #region Innere For-Schleife für das Sammeln aller Optionen (Call und Put)
                for contract_type in contract_types:
                    client.contract_type = contract_type  # 📌 Contract Type wird hier gesetzt

                    if parse_boolean(os.getenv('RUN_FETCH_CONTRACTS')):
                        try:
                            if client.validate_date_range():
                                client.run_fetch_contracts()
                        except ValueError as e:
                            logger.error(f"Error during contract fetching for {underlying_ticker}, type {contract_type}: {e}")
                #endregion

            except Exception as e:
                logger.error(f"Unexpected error for {underlying_ticker}: {e}")
                continue  # Weiter zum nächsten Ticker, falls ein Fehler auftritt

            # 🔹 Hier werden Aggregates erst nach der contract-Schleife gesammelt
            try:
                #region Sammeln von Aggregates
                if parse_boolean(os.getenv('RUN_FETCH_AGGREGATES')):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(run_aggregates(client))
                    loop.close()
                #endregion

                #region Sammeln von Daten für Index
                if parse_boolean(os.getenv('RUN_FETCH_CLOSING_CANDLES')):
                    client.fetch_yfinance_data(
                        ticker=client.index_data_ticker,
                        start_date=client.closing_candles_start_date,
                        end_date=client.closing_candles_end_date,
                        insert_func=client.db_repository.insert_data_of_index
                    )
                #endregion

                #region Sammeln von FRED-Daten
                if parse_boolean(os.getenv('RUN_FETCH_FRED_INTEREST')):
                    series_dict = {
                        "4-Week Treasury Bill": "DTB4WK",
                        "13-Week Treasury Bill": "DTB3",
                        "26-Week Treasury Bill": "DTB6",
                        "52-Week Treasury Bill": "DTB1YR",
                    }
                    for name, series_id in series_dict.items():
                        client.fetch_fred_data(series_id)
                #endregion

                #region Sammeln der Impliziten Volatilität
                if parse_boolean(os.getenv('RUN_FETCH_IMPLIED_VOLATILITY')):
                    client.fetch_yfinance_data(
                        ticker=client.implied_volatility_ticker,
                        start_date=client.implied_volatility_start_date,
                        end_date=client.implied_volatility_end_date,
                        insert_func=client.db_repository.insert_data_implied_volatility
                    )
                #endregion

                ##Performanceindex ^SP500TR von yfinance ziehen

                ##Performanceindex XNDX aus excel ziehen

            except Exception as e:
                logger.error(f"Error during additional fetching steps for {underlying_ticker}: {e}")
            #endregion
        #endregion
    #endregion

    # ----------------------------------------------------------------
    # 3) Aufruf Datenanalyse : Erstellt Prepared Datenbank zum Weiterverarbeiten und führt Auswertung durch
    # ----------------------------------------------------------------

    #region 3) Aufruf Datenverarbeitung + Datenanalyse
    if parse_boolean(os.getenv('RUN_DATA_ANALYSIS')):
        indices = os.getenv("INDICES_TO_PREPARE", "").split(",")

        index_data_mapping = parse_env_mapping("INDEX_DATA_MAPPING")

        for index in indices:
            print(f"🔄 Starte Verarbeitung für Index: {index}")

            index_ticker = index_data_mapping.get(index, None)

            # 📌 **Korrekter Datenbanknamen für die Raw-Daten**
            raw_db_name = f"rawdata_{index.lower()}_db.sqlite"

            # ----------------------------------------------------------------
            # 3.1) Aufruf Datenvorbereitung
            # ----------------------------------------------------------------

            #region 3.1) Aufruf Datenvorbereitung: Vorbereitung für Datenanalyse
            if parse_boolean(os.getenv('RUN_DATA_HANDLER')):
                print(f"📊 Running DataHandler für {index}...")

                data_handler = DataHandler(
                    raw_database_path=os.getenv("RAW_DATABASE_PATH"),
                    raw_db_names=[raw_db_name],  # Einzelne DB pro Iteration
                    sorted_db_path=os.path.join(os.getenv("DB_ANALYSIS_PATH"), os.getenv("DB_ANALYSIS_FILENAME")),
                    batch_size=int(os.getenv("ANALYSIS_BATCH_SIZE"))
                )

                db_repository = get_analysis_database_repository()
                db_repository.analysis_db_migrate()  # Stellt sicher, dass alle Tabellen existieren
                data_handler.process_contracts(db_repository)
                db_repository.close()

            # ✅ **DataPreparer: Prepare Data for further calculations**
            if parse_boolean(os.getenv('RUN_DATA_PREPARER')):
                print(f"📈 Running DataPreparer für {index}...")

                data_preparer = DataPreparer(
                    sorted_db_path=os.path.join(os.getenv("SORTED_DB_PATH"), os.getenv("SORTED_DB_FILENAME")),
                    prepared_db_path=os.path.join(os.getenv("PREPARED_DB_PATH"), os.getenv("PREPARED_DB_FILENAME")),
                    indices_to_prepare=os.getenv("INDICES_TO_PREPARE"),
                    index=index
                )

                #Kopiert die ursprüngliche sorted Datenbank, falls noch nicht vorhanden
                data_preparer.initialize_prepared_db()
                data_preparer.fetch_yfinance_data(
                    index_ticker,
                    os.getenv("CLOSING_CANDLES_START_DATE"),
                    os.getenv("CLOSING_CANDLES_END_DATE")
                )

                #Konvertiert Volatilität von [%] zu Dezimal
                if parse_boolean(os.getenv('RUN_CONVERT_VOLA')):
                    data_preparer.convert_implied_volatility()

                #Berechnet dividend_yield
                if parse_boolean(os.getenv('RUN_DIVIDEND_YIELD')):
                    data_preparer.calc_dividend_yield()
            #endregion

            # ----------------------------------------------------------------
            # 3.2) Aufruf Datenanalyse
            # ----------------------------------------------------------------

            #region 3.2) Aufruf Datenanalyse
            if parse_boolean(os.getenv('RUN_DATA_ANALYZER')):
                print(f"📈 Running DataAnalyzer für {index}...")

                data_analyzer = DataAnalyzer(
                    sorted_db_path=os.path.join(os.getenv("DB_ANALYSIS_PATH"), os.getenv("DB_ANALYSIS_FILENAME")),
                    indices_to_analyze=[index]  # Einzelne Index-Analyse pro Durchlauf
                )

                data_analyzer.analyze_and_plot()
            #endregion
        print("🎉 Alle Indizes wurden erfolgreich verarbeitet!")
    #endregion


if __name__ == '__main__':
    main()
