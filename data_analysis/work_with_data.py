#Ermittlung Dividendenrendite, diese sollte stimmen und entsprechend keine Prio sein, ich habe immer mindestens 4 Zeilen Abstand gelassen wenn ein neues Code beginnt (zur Orientierung)
#Wenn du den Fokus auf Erstellung BSM-Preis legen könntest wäre das genial
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Datenbankpfade für Optionen
db_ndx_options = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"
db_spx_options = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"

# Datenbankpfade für Preisindizes
db_ndx_index = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/bulli-project/data/data_ndx_call_and_put_db.sqlite"
db_spx_index = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/bulli-project/data/data_spx_call_and_put_db.sqlite"

# Excel-Dateipfade für Performanceindizes
excel_ndx = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/XNDX.xlsx"
excel_spx = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/SP500TR.xlsx"

# Tabellen & Spaltennamen
ndx_options_table = "sorted_ndx_data"
spx_options_table = "sorted_spx_data"
index_table = "index_data"
date_col_options = "trade_date"  # In den Optionsdaten
date_col_index = "date"  # In den Indexdaten
price_col = "close"

# Frühestes verfügbares Datum für Performanceindex
min_date = datetime(2017, 1, 3)


def find_nearest_previous_date(target_date, available_dates):
    available_dates = sorted(pd.to_datetime(available_dates))
    if target_date < min_date:
        return min_date  # Falls das Datum vor dem 03.01.2017 liegt, nutze 03.01.2017
    previous_dates = [d for d in available_dates if d <= target_date]
    return previous_dates[-1] if previous_dates else min_date


# Funktion zur Berechnung der Dividendenrendite

def calculate_dividend_yield(trade_date, index_dates, index_data, perf_index_data):
    one_year_ago = trade_date - timedelta(days=365)

    nearest_date_t = find_nearest_previous_date(trade_date, index_dates)
    nearest_date_t1y = find_nearest_previous_date(one_year_ago, index_dates)

    if nearest_date_t is None or nearest_date_t1y is None:
        print(f"❌ Kein gültiges Datum für {trade_date.strftime('%Y-%m-%d')} gefunden!")
        return None

    # Stelle sicher, dass die Datumsspalte datetime ist
    index_data[date_col_index] = pd.to_datetime(index_data[date_col_index])

    # Versuche, die Preisindexwerte zu finden
    price_index_t = index_data.loc[index_data[date_col_index] == nearest_date_t, price_col]
    price_index_t1y = index_data.loc[index_data[date_col_index] == nearest_date_t1y, price_col]

    if price_index_t.empty or price_index_t1y.empty:
        print(
            f"❌ Kein Preisindex für {nearest_date_t.strftime('%Y-%m-%d')} oder {nearest_date_t1y.strftime('%Y-%m-%d')} gefunden!")
        return None

    price_index_t = price_index_t.values[0]
    price_index_t1y = price_index_t1y.values[0]

    if nearest_date_t in perf_index_data.index and nearest_date_t1y in perf_index_data.index:
        perf_index_t = perf_index_data.loc[nearest_date_t].values[0]
        perf_index_t1y = perf_index_data.loc[nearest_date_t1y].values[0]

        return ((perf_index_t / perf_index_t1y) / (price_index_t / price_index_t1y)) - 1

    return None


# Funktion zur Berechnung und Speicherung der Dividendenrendite für alle Optionen eines Tages

def process_dividend_yield(db_options, db_index, excel_file, options_table):
    # Verbindung zur Options-Datenbank
    conn_options = sqlite3.connect(db_options)
    trade_dates = pd.read_sql(f"SELECT DISTINCT {date_col_options} FROM {options_table}", conn_options)[
        date_col_options]

    # Verbindung zur Index-Datenbank
    conn_index = sqlite3.connect(db_index)
    index_data = pd.read_sql(f"SELECT {date_col_index}, {price_col} FROM {index_table}", conn_index)
    index_dates = index_data[date_col_index]
    conn_index.close()

    # Performance-Daten laden
    perf_index_data = pd.read_excel(excel_file, sheet_name=0)
    perf_index_data[date_col_index] = pd.to_datetime(perf_index_data[date_col_index])
    perf_index_data = perf_index_data.set_index(date_col_index)

    cursor = conn_options.cursor()

    for trade_date in trade_dates:
        trade_date = pd.to_datetime(trade_date)
        dividend_yield = calculate_dividend_yield(trade_date, index_dates, index_data, perf_index_data)

        if dividend_yield is not None:
            cursor.execute(f"""
                UPDATE {options_table}
                SET dividend_yield = ?
                WHERE {date_col_options} = ?
            """, (dividend_yield, trade_date.strftime('%Y-%m-%d')))

            # ✅ Hier die Bestätigung ausgeben:
            print(
                f"✅ {trade_date.strftime('%Y-%m-%d')}: Dividendenrendite berechnet und gespeichert ({dividend_yield:.6f})")

    conn_options.commit()
    conn_options.close()


# Berechnung und Speicherung für beide Märkte
process_dividend_yield(db_ndx_options, db_ndx_index, excel_ndx, ndx_options_table)
process_dividend_yield(db_spx_options, db_spx_index, excel_spx, spx_options_table)

print("✅ Alle Dividendenrenditen wurden berechnet und in der Datenbank aktualisiert!")




#Berechnung BSM, Hier habe ich das Problem, dass es eine Formel für Put und für Call-Optionen gibt und das einfach nich funktioniert diese richtig zu erkennen
# Ich sende dir auch noch eine Word Datei in der die Black Scholes Formel mit Einheit der Input-Parameter beschrieben werden

from scipy.stats import norm

# Datenbankpfade
db_ndx_options = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"
db_spx_options = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"

# Tabellen mit Optionsdaten
ndx_options_table = "sorted_ndx_data"
spx_options_table = "sorted_spx_data"


# Black-Scholes-Merton Formel mit und ohne Dividendenrendite
def black_scholes_merton(S0, K, T, r, q, sigma, option_type):
    """Berechnet den BSM-Preis für eine Call- oder Put-Option mit oder ohne konstante Dividendenrendite."""
    if T <= 0 or sigma <= 0:
        return None  # Falls ungültige Werte auftreten

    # Falls `q` NULL ist, setzen wir sie auf 0 (Standard-Black-Scholes-Modell)
    if q is None:
        q = 0

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put-Option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)

    return price


# Funktion zur Erkennung des Optionstyps aus dem Ticker (Call oder Put)
def get_option_type(ticker):
    """
    Identifiziert, ob der Ticker eine Call- oder Put-Option ist.
    Stellt sicher, dass NDXP und SPXW nicht fälschlicherweise als Put-Optionen erkannt werden.
    """
    match = re.search(r'(NDX|SPX)([A-Z]?)(\d{6})([CP])', ticker)
    if match:
        option_type = match.group(4)  # "C" für Call, "P" für Put
        return "call" if option_type == "C" else "put"

    return None  # Falls kein gültiges Muster gefunden wurde


# Lade Optionsdaten aus der Datenbank (inkl. Ticker)
def load_options_data(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"""
            SELECT rowid, ticker, execution_price AS K, 
                   market_price_base AS S0, 
                   remaining_time, 
                   risk_free_interest AS r, 
                   implied_volatility AS sigma, 
                   dividend_yield AS q
            FROM {table_name}
        """, conn)

    # Umwandlung der Werte:
    df["remaining_time"] = df["remaining_time"] / 365  # Tage -> Jahre
    df["sigma"] = df["sigma"] / 100  # Prozent -> Dezimalzahl

    # Falls die Dividendenrendite `NULL` ist, wird sie in Pandas als `NaN` erkannt, daher setzen wir sie explizit auf `None`
    df["q"] = df["q"].where(pd.notna(df["q"]), None)

    # Optionstyp aus dem Ticker extrahieren
    df["option_type"] = df["ticker"].apply(get_option_type)

    return df


# Funktion zum Aktualisieren der berechneten BSM-Preise in der Datenbank
def update_database(db_path, table_name, updates):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(f"""
            UPDATE {table_name}
            SET BSM_price = ?
            WHERE rowid = ?
        """, updates)
        conn.commit()


# Berechnung und Speicherung für Nasdaq 100 & S&P 500
def process_options(db_path, table_name):
    options = load_options_data(db_path, table_name)
    updates = []

    for i, row in options.iterrows():
        rowid, ticker, K, S0, T, r, sigma, q, option_type = row  # Korrekte Reihenfolge!

        if option_type is None:
            print(f"⚠️ Optionstyp für {ticker} konnte nicht bestimmt werden. Übersprungen.")
            continue

        bsm_price = black_scholes_merton(S0, K, T, r, q, sigma, option_type)  # Preis berechnen
        if bsm_price is not None:
            updates.append((bsm_price, rowid))

    update_database(db_path, table_name, updates)


# Berechnung für beide Tabellen
process_options(db_ndx_options, ndx_options_table)
process_options(db_spx_options, spx_options_table)

print("\n✅ Alle berechneten BSM-Preise wurden erfolgreich in die Datenbank gespeichert!")






#Abweichungen ermitteln, dieser hier funktioniert der Fehler liegt davor
import sqlite3
import pandas as pd

# Datenbankpfade
db_ndx_options = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"
db_spx_options = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"

# Tabellen mit Optionsdaten
ndx_options_table = "sorted_ndx_data"
spx_options_table = "sorted_spx_data"


# Funktion zum Hinzufügen der Spalten `absolute_error` und `relative_error_percent`, falls sie nicht existieren
def add_columns_if_not_exist(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]

        if "absolute_error" not in columns:
            print(f"⚙️ Spalte 'absolute_error' wird zu {table_name} hinzugefügt...")
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN absolute_error REAL")

        if "relative_error_percent" not in columns:  # Korrektur: Kein "%" im Spaltennamen
            print(f"⚙️ Spalte 'relative_error_percent' wird zu {table_name} hinzugefügt...")
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN relative_error_percent REAL")

        conn.commit()


# Spalten zu beiden Tabellen hinzufügen, falls nicht vorhanden
add_columns_if_not_exist(db_ndx_options, ndx_options_table)
add_columns_if_not_exist(db_spx_options, spx_options_table)


# Funktion zum Laden der Optionsdaten mit Markt- und Modellpreis
def load_options_data(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"""
            SELECT rowid, market_price_option AS P_Markt, BSM_price AS P_Modell
            FROM {table_name}
        """, conn)
    return df


# Lade Optionsdaten für Nasdaq 100 & S&P 500
ndx_options = load_options_data(db_ndx_options, ndx_options_table)
spx_options = load_options_data(db_spx_options, spx_options_table)


# Funktion zur Berechnung der Fehlerwerte und Aktualisierung der Datenbank
def calculate_and_update_errors(db_path, table_name, df):
    updates = []

    for i, row in df.iterrows():
        rowid, P_Markt, P_Modell = row

        if P_Modell is None or P_Markt is None:
            continue  # Überspringe, falls fehlende Werte vorliegen

        absolute_error = abs(P_Modell - P_Markt)
        relative_error = (absolute_error / P_Markt) * 100 if P_Markt != 0 else None  # Schutz gegen Division durch 0

        updates.append((absolute_error, relative_error, rowid))

    # Aktualisiere die Datenbank mit den berechneten Fehlern
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(f"""
            UPDATE {table_name}
            SET absolute_error = ?, relative_error_percent = ?
            WHERE rowid = ?
        """, updates)
        conn.commit()


# Berechnung und Speicherung für Nasdaq 100 & S&P 500
calculate_and_update_errors(db_ndx_options, ndx_options_table, ndx_options)
calculate_and_update_errors(db_spx_options, spx_options_table, spx_options)

print("\n✅ Alle berechneten Abweichungen wurden erfolgreich in die Datenbank gespeichert!")







#Analyse der Daten, Abweichung absolut und relativ über die Zeit, funktioniert auch
#Ab hier kommen viele Diagramme von denen aber auch manche die Klassifizierung für Put und Call enthalten weshalb da vermutlich ebenfalls Fehler drin sind
# Ich sende dir eine Word Datei mit denen, die ich bisher erstellt ahbe, damit du verstehst was ich machen will un denen die noch fehlen

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Datenbankpfade
ndx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"
spx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"

# SQL-Abfrage zum Abrufen der relevanten Daten
query = """
    SELECT trade_date, relative_error_percent, absolute_error 
    FROM sorted_ndx_data
"""

# Verbindung zur Datenbank herstellen und Daten abrufen
def load_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT trade_date, relative_error_percent, absolute_error FROM {table_name}", conn)
    conn.close()
    return df

# Daten laden
ndx_data = load_data(ndx_db_path, "sorted_ndx_data")
spx_data = load_data(spx_db_path, "sorted_spx_data")

# Datumsformatierung
ndx_data['trade_date'] = pd.to_datetime(ndx_data['trade_date'])
spx_data['trade_date'] = pd.to_datetime(spx_data['trade_date'])

# Monatliche Aggregation
ndx_monthly = ndx_data.resample('ME', on='trade_date').mean()
spx_monthly = spx_data.resample('ME', on='trade_date').mean()

# Liniendiagramm erstellen
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

# Relative Abweichung
axes[0].plot(ndx_monthly.index, ndx_monthly['relative_error_percent'], label='Nasdaq 100', color='navy')
axes[0].plot(spx_monthly.index, spx_monthly['relative_error_percent'], label='S&P 500', color='deepskyblue')
axes[0].set_title("Monatliche durchschnittliche relative Abweichung")
axes[0].set_ylabel("Relative Abweichung (%)")
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.7)

# Absolute Abweichung
axes[1].plot(ndx_monthly.index, ndx_monthly['absolute_error'], label='Nasdaq 100', color='navy')
axes[1].plot(spx_monthly.index, spx_monthly['absolute_error'], label='S&P 500', color='deepskyblue')
axes[1].set_title("Monatliche durchschnittliche absolute Abweichung")
axes[1].set_ylabel("Absolute Abweichung (USD)")
axes[1].set_xlabel("Zeit")
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.7)

# X-Achse formatieren (nicht jeden Monat beschriften)
axes[1].xaxis.set_major_locator(mdates.YearLocator())  # Nur Jahresmarkierungen
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()




#Weitere Diagramme

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Datenbankpfade
ndx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"
spx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"

# Verbindung zur Datenbank herstellen und Daten abrufen
def load_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT trade_date, relative_error_percent, remaining_time, market_price_base, execution_price, ticker FROM {table_name}", conn)
    conn.close()
    return df

# Daten laden
ndx_data = load_data(ndx_db_path, "sorted_ndx_data")
spx_data = load_data(spx_db_path, "sorted_spx_data")

# Datumsformatierung
ndx_data['trade_date'] = pd.to_datetime(ndx_data['trade_date'])
spx_data['trade_date'] = pd.to_datetime(spx_data['trade_date'])

# Optionstyp bestimmen
def identify_option_type(ticker):
    if pd.isna(ticker):
        return None
    match = re.search(r'[CP]', ticker)
    if match:
        return 'Call' if match.group() == 'C' else 'Put'
    return None

ndx_data['option_type'] = ndx_data['ticker'].apply(identify_option_type)
spx_data['option_type'] = spx_data['ticker'].apply(identify_option_type)

# Korrekte Berechnung der Moneyness
ndx_data['moneyness'] = np.where(ndx_data['option_type'] == 'Call',
                                 ndx_data['market_price_base'] / ndx_data['execution_price'],
                                 ndx_data['execution_price'] / ndx_data['market_price_base'])

spx_data['moneyness'] = np.where(spx_data['option_type'] == 'Call',
                                 spx_data['market_price_base'] / spx_data['execution_price'],
                                 spx_data['execution_price'] / spx_data['market_price_base'])

# Moneyness-Kategorien definieren
ndx_data['moneyness_category'] = pd.cut(ndx_data['moneyness'], bins=[0, 0.99, 1.01, np.inf], labels=['im Geld', 'am Geld', 'aus dem Geld'])
spx_data['moneyness_category'] = pd.cut(spx_data['moneyness'], bins=[0, 0.99, 1.01, np.inf], labels=['im Geld', 'am Geld', 'aus dem Geld'])

# Laufzeitgruppen definieren
bins = [0, 30, 90, np.inf]
labels = ['0-30 Tage', '31-90 Tage', '>90 Tage']
ndx_data['remaining_time_group'] = pd.cut(ndx_data['remaining_time'], bins=bins, labels=labels)
spx_data['remaining_time_group'] = pd.cut(spx_data['remaining_time'], bins=bins, labels=labels)

# Balkendiagramm für Laufzeitgruppen
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(labels))
width = 0.35
ndx_laufzeit_avg = ndx_data.groupby('remaining_time_group', observed=False)['relative_error_percent'].mean()
spx_laufzeit_avg = spx_data.groupby('remaining_time_group', observed=False)['relative_error_percent'].mean()

ax.bar(x - width/2, ndx_laufzeit_avg, width, label='Nasdaq 100', color='navy')
ax.bar(x + width/2, spx_laufzeit_avg, width, label='S&P 500', color='deepskyblue')
ax.set_xlabel("Laufzeitgruppen")
ax.set_ylabel("Durchschnittliche relative Abweichung (%)")
ax.set_title("Durchschnittliche relative Abweichung nach Laufzeitgruppen")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Balkendiagramm für Moneyness-Kategorien
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(3)
width = 0.35
ndx_moneyness_avg = ndx_data.groupby('moneyness_category', observed=False)['relative_error_percent'].mean()
spx_moneyness_avg = spx_data.groupby('moneyness_category', observed=False)['relative_error_percent'].mean()

ax.bar(x - width/2, ndx_moneyness_avg, width, label='Nasdaq 100', color='navy')
ax.bar(x + width/2, spx_moneyness_avg, width, label='S&P 500', color='deepskyblue')
ax.set_xlabel("Moneyness-Kategorien")
ax.set_ylabel("Durchschnittliche relative Abweichung (%)")
ax.set_title("Durchschnittliche relative Abweichung nach Moneyness")
ax.set_xticks(x)
ax.set_xticklabels(['im Geld', 'am Geld', 'aus dem Geld'])
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Liniendiagramm für die monatliche Entwicklung
ndx_monthly = ndx_data.groupby([pd.Grouper(key='trade_date', freq='ME'), 'remaining_time_group'], observed=False)['relative_error_percent'].mean().unstack()
spx_monthly = spx_data.groupby([pd.Grouper(key='trade_date', freq='ME'), 'remaining_time_group'], observed=False)['relative_error_percent'].mean().unstack()

fig, ax = plt.subplots(figsize=(12, 6))
ndx_monthly.plot(ax=ax, colormap='coolwarm')
ax.set_title("Monatliche Entwicklung der relativen Abweichung - Nasdaq 100")
ax.set_xlabel("Zeit")
ax.set_ylabel("Relative Abweichung (%)")
ax.legend(title="Laufzeitgruppe")
ax.grid(True, linestyle='--', alpha=0.7)
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
spx_monthly.plot(ax=ax, colormap='coolwarm')
ax.set_title("Monatliche Entwicklung der relativen Abweichung - S&P 500")
ax.set_xlabel("Zeit")
ax.set_ylabel("Relative Abweichung (%)")
ax.legend(title="Laufzeitgruppe")
ax.grid(True, linestyle='--', alpha=0.7)
plt.show()


#Moneyness-Diagramme

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Datenbankpfade
ndx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"
spx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"


# Verbindung zur Datenbank herstellen und Daten abrufen
def load_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT trade_date, relative_error_percent, remaining_time, market_price_base, execution_price, ticker FROM {table_name}",
        conn)
    conn.close()
    return df


# Daten laden
ndx_data = load_data(ndx_db_path, "sorted_ndx_data")
spx_data = load_data(spx_db_path, "sorted_spx_data")

# Datumsformatierung
ndx_data['trade_date'] = pd.to_datetime(ndx_data['trade_date'])
spx_data['trade_date'] = pd.to_datetime(spx_data['trade_date'])


# Optionstyp bestimmen (Korrektur der Erkennung)
def identify_option_type(ticker):
    """
    Identifiziert, ob der Ticker eine Call- oder Put-Option ist.
    Stellt sicher, dass NDXP und SPXW nicht fälschlicherweise als Put-Optionen erkannt werden.
    """
    match = re.search(r'(NDX|SPX)[A-Z]?(\d{6})([CP])', ticker)
    if match:
        option_type = match.group(3)  # "C" für Call, "P" für Put
        return "Call" if option_type == "C" else "Put"
    return None  # Falls kein gültiges Muster gefunden wurde


ndx_data['option_type'] = ndx_data['ticker'].apply(identify_option_type)
spx_data['option_type'] = spx_data['ticker'].apply(identify_option_type)

# Debugging: Optionstyp überprüfen
print(ndx_data[['ticker', 'option_type']].head(20))

# Korrekte Berechnung der Moneyness
ndx_data['moneyness'] = np.where(ndx_data['option_type'] == 'Call',
                                 ndx_data['execution_price'] / ndx_data['market_price_base'],
                                 ndx_data['market_price_base'] / ndx_data['execution_price'])

spx_data['moneyness'] = np.where(spx_data['option_type'] == 'Call',
                                 spx_data['execution_price'] / spx_data['market_price_base'],
                                 spx_data['market_price_base'] / spx_data['execution_price'])

# Debugging: Moneyness überprüfen
print(ndx_data[['moneyness']].value_counts())

# Moneyness-Kategorien definieren
ndx_data['moneyness_category'] = pd.cut(ndx_data['moneyness'], bins=[0, 0.97, 1.03, np.inf],
                                        labels=['im Geld', 'am Geld', 'aus dem Geld'])
spx_data['moneyness_category'] = pd.cut(spx_data['moneyness'], bins=[0, 0.97, 1.03, np.inf],
                                        labels=['im Geld', 'am Geld', 'aus dem Geld'])

# Laufzeitgruppen definieren
bins = [0, 30, 90, np.inf]
labels = ['0-30 Tage', '31-90 Tage', '>90 Tage']
ndx_data['remaining_time_group'] = pd.cut(ndx_data['remaining_time'], bins=bins, labels=labels)
spx_data['remaining_time_group'] = pd.cut(spx_data['remaining_time'], bins=bins, labels=labels)

# Farben für die Laufzeitgruppen
colors = ['navy', 'royalblue', 'deepskyblue']


# Funktion zur Diagrammerstellung
def plot_moneyness_chart(data, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(['im Geld', 'am Geld', 'aus dem Geld']))
    width = 0.25

    for i, label in enumerate(labels):
        means = [
            data[(data['moneyness_category'] == category) & (data['remaining_time_group'] == label)][
                'relative_error_percent'].mean()
            for category in ['im Geld', 'am Geld', 'aus dem Geld']
        ]
        ax.bar(x + i * width - width, means, width, label=f'{label}', color=colors[i])

    ax.set_xlabel("Moneyness-Kategorie")
    ax.set_ylabel("Durchschnittliche relative Abweichung (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(['im Geld', 'am Geld', 'aus dem Geld'])
    ax.legend(title="Laufzeitgruppe")
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# Diagramme erstellen
plot_moneyness_chart(ndx_data, "Durchschnittliche relative Abweichung - Nasdaq 100")
plot_moneyness_chart(spx_data, "Durchschnittliche relative Abweichung - S&P 500")







#Moneyness allgemein
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Datenbankpfade
ndx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"
spx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"

# Verbindung zur Datenbank herstellen und Daten abrufen
def load_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT trade_date, relative_error_percent, remaining_time, market_price_base, execution_price, ticker FROM {table_name}", conn)
    conn.close()
    return df

# Daten laden
ndx_data = load_data(ndx_db_path, "sorted_ndx_data")
spx_data = load_data(spx_db_path, "sorted_spx_data")

# Datumsformatierung
ndx_data['trade_date'] = pd.to_datetime(ndx_data['trade_date'])
spx_data['trade_date'] = pd.to_datetime(spx_data['trade_date'])

# Optionstyp bestimmen (Korrektur der Erkennung)
def identify_option_type(ticker):
    """
    Identifiziert, ob der Ticker eine Call- oder Put-Option ist.
    Stellt sicher, dass NDXP und SPXW nicht fälschlicherweise als Put-Optionen erkannt werden.
    """
    match = re.search(r'(NDX|SPX)[A-Z]?(\d{6})([CP])', ticker)
    if match:
        option_type = match.group(3)  # "C" für Call, "P" für Put
        return "Call" if option_type == "C" else "Put"
    return None  # Falls kein gültiges Muster gefunden wurde

ndx_data['option_type'] = ndx_data['ticker'].apply(identify_option_type)
spx_data['option_type'] = spx_data['ticker'].apply(identify_option_type)

# Korrekte Berechnung der Moneyness
ndx_data['moneyness'] = np.where(ndx_data['option_type'] == 'Call',
                                 ndx_data['execution_price'] / ndx_data['market_price_base'],
                                 ndx_data['market_price_base'] / ndx_data['execution_price'])

spx_data['moneyness'] = np.where(spx_data['option_type'] == 'Call',
                                 spx_data['execution_price'] / spx_data['market_price_base'],
                                 spx_data['market_price_base'] / spx_data['execution_price'])

# Moneyness-Kategorien definieren
ndx_data['moneyness_category'] = pd.cut(ndx_data['moneyness'], bins=[0, 0.97, 1.03, np.inf], labels=['im Geld', 'am Geld', 'aus dem Geld'])
spx_data['moneyness_category'] = pd.cut(spx_data['moneyness'], bins=[0, 0.97, 1.03, np.inf], labels=['im Geld', 'am Geld', 'aus dem Geld'])

# Berechnung der durchschnittlichen relativen Abweichung nach Moneyness
ndx_moneyness_avg = ndx_data.groupby('moneyness_category', observed=False)['relative_error_percent'].mean()
spx_moneyness_avg = spx_data.groupby('moneyness_category', observed=False)['relative_error_percent'].mean()

# Balkendiagramm für Moneyness-Kategorien
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(ndx_moneyness_avg.index))
width = 0.35

ax.bar(x - width/2, ndx_moneyness_avg, width, label='Nasdaq 100', color='navy')
ax.bar(x + width/2, spx_moneyness_avg, width, label='S&P 500', color='deepskyblue')

ax.set_xlabel("Moneyness-Kategorien")
ax.set_ylabel("Durchschnittliche relative Abweichung (%)")
ax.set_title("Durchschnittliche relative Abweichung nach Moneyness")
ax.set_xticks(x)
ax.set_xticklabels(ndx_moneyness_avg.index)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

plt.show()

#Verteilung Moneyness Lafzeiten bei OTM
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Datenbankpfade
ndx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"
spx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"

# Verbindung zur Datenbank herstellen und Daten abrufen
def load_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT trade_date, relative_error_percent, remaining_time, market_price_base, execution_price, ticker FROM {table_name}", conn)
    conn.close()
    return df

# Daten laden
ndx_data = load_data(ndx_db_path, "sorted_ndx_data")
spx_data = load_data(spx_db_path, "sorted_spx_data")

# Datumsformatierung
ndx_data['trade_date'] = pd.to_datetime(ndx_data['trade_date'])
spx_data['trade_date'] = pd.to_datetime(spx_data['trade_date'])

# Optionstyp bestimmen (Korrektur der Erkennung)
def identify_option_type(ticker):
    """
    Identifiziert, ob der Ticker eine Call- oder Put-Option ist.
    Stellt sicher, dass NDXP und SPXW nicht fälschlicherweise als Put-Optionen erkannt werden.
    """
    match = re.search(r'(NDX|SPX)[A-Z]?(\d{6})([CP])', ticker)
    if match:
        option_type = match.group(3)  # "C" für Call, "P" für Put
        return "Call" if option_type == "C" else "Put"
    return None  # Falls kein gültiges Muster gefunden wurde

ndx_data['option_type'] = ndx_data['ticker'].apply(identify_option_type)
spx_data['option_type'] = spx_data['ticker'].apply(identify_option_type)

# Korrekte Berechnung der Moneyness
ndx_data['moneyness'] = np.where(ndx_data['option_type'] == 'Call',
                                 ndx_data['execution_price'] / ndx_data['market_price_base'],
                                 ndx_data['market_price_base'] / ndx_data['execution_price'])

spx_data['moneyness'] = np.where(spx_data['option_type'] == 'Call',
                                 spx_data['execution_price'] / spx_data['market_price_base'],
                                 spx_data['market_price_base'] / spx_data['execution_price'])

# Moneyness-Kategorien definieren (fix für fehlende Spalte)
ndx_data['moneyness_category'] = pd.cut(ndx_data['moneyness'], bins=[0, 0.97, 1.03, np.inf], labels=['im Geld', 'am Geld', 'aus dem Geld'])
spx_data['moneyness_category'] = pd.cut(spx_data['moneyness'], bins=[0, 0.97, 1.03, np.inf], labels=['im Geld', 'am Geld', 'aus dem Geld'])

# Laufzeitgruppen definieren (Fix für fehlende Spalte)
bins = [0, 30, 90, np.inf]
labels = ['0-30 Tage', '31-90 Tage', '>90 Tage']
ndx_data['remaining_time_group'] = pd.cut(ndx_data['remaining_time'], bins=bins, labels=labels)
spx_data['remaining_time_group'] = pd.cut(spx_data['remaining_time'], bins=bins, labels=labels)

# Filter auf Out-of-the-Money-Optionen
ndx_otm = ndx_data[ndx_data['moneyness_category'] == 'aus dem Geld']
spx_otm = spx_data[spx_data['moneyness_category'] == 'aus dem Geld']

# Berechnung der durchschnittlichen Moneyness für jede Laufzeitgruppe
ndx_deepness_avg = ndx_otm.groupby('remaining_time_group', observed=False)['moneyness'].mean()
spx_deepness_avg = spx_otm.groupby('remaining_time_group', observed=False)['moneyness'].mean()

# Farben für die Laufzeitgruppen
colors = ['navy', 'royalblue', 'deepskyblue']

# Balkendiagramm für Deepness-Wert
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(ndx_deepness_avg.index))
width = 0.35

ax.bar(x - width/2, ndx_deepness_avg, width, label='Nasdaq 100', color=colors[0])
ax.bar(x + width/2, spx_deepness_avg, width, label='S&P 500', color=colors[1])

ax.set_xlabel("Laufzeitgruppe")
ax.set_ylabel("Durchschnittliche Moneyness (%)")
ax.set_title("Durchschnittliche Moneyness nach Laufzeitgruppen")
ax.set_xticks(x)
ax.set_xticklabels(ndx_deepness_avg.index)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

plt.show()











#Abhängigkeit Moneyness relative error

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Datenbankpfade
ndx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"
spx_db_path = "C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/analyze_option_data/data/sorted_option_data_db.sqlite"

# Verbindung zur Datenbank herstellen und Daten abrufen
def load_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT trade_date, relative_error_percent, remaining_time, market_price_base, execution_price, ticker FROM {table_name}", conn)
    conn.close()
    return df

# Daten laden
ndx_data = load_data(ndx_db_path, "sorted_ndx_data")
spx_data = load_data(spx_db_path, "sorted_spx_data")

# Datumsformatierung
ndx_data['trade_date'] = pd.to_datetime(ndx_data['trade_date'])
spx_data['trade_date'] = pd.to_datetime(spx_data['trade_date'])

# Optionstyp bestimmen (Korrektur der Erkennung)
def identify_option_type(ticker):
    """
    Identifiziert, ob der Ticker eine Call- oder Put-Option ist.
    Stellt sicher, dass NDXP und SPXW nicht fälschlicherweise als Put-Optionen erkannt werden.
    """
    match = re.search(r'(NDX|SPX)[A-Z]?(\d{6})([CP])', ticker)
    if match:
        option_type = match.group(3)  # "C" für Call, "P" für Put
        return "Call" if option_type == "C" else "Put"
    return None  # Falls kein gültiges Muster gefunden wurde

ndx_data['option_type'] = ndx_data['ticker'].apply(identify_option_type)
spx_data['option_type'] = spx_data['ticker'].apply(identify_option_type)

# Korrekte Berechnung der Moneyness
ndx_data['moneyness'] = np.where(ndx_data['option_type'] == 'Call',
                                 ndx_data['execution_price'] / ndx_data['market_price_base'],
                                 ndx_data['market_price_base'] / ndx_data['execution_price'])

spx_data['moneyness'] = np.where(spx_data['option_type'] == 'Call',
                                 spx_data['execution_price'] / spx_data['market_price_base'],
                                 spx_data['market_price_base'] / spx_data['execution_price'])

# Moneyness-Kategorien definieren
ndx_data['moneyness_category'] = pd.cut(ndx_data['moneyness'], bins=[0, 0.97, 1.03, np.inf], labels=['im Geld', 'am Geld', 'aus dem Geld'])
spx_data['moneyness_category'] = pd.cut(spx_data['moneyness'], bins=[0, 0.97, 1.03, np.inf], labels=['im Geld', 'am Geld', 'aus dem Geld'])

# Streudiagramm für die relative Abweichung in Abhängigkeit von der Moneyness
def plot_moneyness_vs_error(data, title):
    # Filter für Out-of-the-Money-Optionen
    data = data[data['moneyness_category'] == 'aus dem Geld']
    plt.figure(figsize=(10, 5))
    plt.scatter(data['moneyness'] * 100, data['relative_error_percent'], alpha=0.5, color='navy')
    plt.xlabel("Moneyness (%)")
    plt.ylabel("Relative Abweichung (%)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Diagramme erstellen
plot_moneyness_vs_error(ndx_data, "Relative Abweichung in Abhängigkeit von der Moneyness - Nasdaq 100")
plot_moneyness_vs_error(spx_data, "Relative Abweichung in Abhängigkeit von der Moneyness - S&P 500")
