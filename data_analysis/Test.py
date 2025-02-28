import sqlite3
import numpy as np
from scipy.stats import norm
import re

def add_bsm_price_column(conn, table_name):
    """Fügt die Spalte 'BSM_Price' hinzu, falls sie nicht existiert."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor.fetchall()]

    if "BSM_Price" not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN BSM_Price REAL")
        print(f"✅ Spalte 'BSM_Price' zu {table_name} hinzugefügt.")
    else:
        print(f"ℹ️ Spalte 'BSM_Price' existiert bereits in {table_name}.")


def black_scholes_price(S, K, T, r, sigma, option_type):
    """Berechnet den Black-Scholes-Preis für Call- oder Put-Optionen."""
    if T <= 0 or sigma <= 0:
        return 0.0  # Keine Laufzeit oder Volatilität -> Preis 0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "Put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return None


def calculate_and_update_bsm_prices(conn, table_name, batch_size=1000):
    """Berechnet BSM-Preise für alle Optionen und speichert sie in der Datenbank."""
    cursor = conn.cursor()

    # Alle Optionen laden
    query = f"""
    SELECT id, ticker, execution_price, market_price_base, remaining_time, risk_free_interest, implied_volatility
    FROM {table_name}
    WHERE execution_price IS NOT NULL AND market_price_base IS NOT NULL AND remaining_time IS NOT NULL 
          AND risk_free_interest IS NOT NULL AND implied_volatility IS NOT NULL
    """
    cursor.execute(query)
    options = cursor.fetchall()

    print(f"\n📊 Starte Berechnung für {len(options)} Optionen in {table_name}...\n")

    updates = []
    for i, (option_id, ticker, K, S, T, r, sigma) in enumerate(options, start=1):
        T_years = T / 365
        sigma_decimal = sigma / 100  # Volatilität umwandeln

        # Optionstyp erkennen
        if re.search(r'C\d{6}', ticker.upper()):
            option_type = "Call"
        elif re.search(r'P\d{6}', ticker.upper()):
            option_type = "Put"
        else:
            continue  # Unbekannter Typ überspringen

        # Black-Scholes-Preis berechnen
        bsm_price = black_scholes_price(S, K, T_years, r, sigma_decimal, option_type)
        updates.append((bsm_price, option_id))

        # Fortschrittsanzeige alle 1000 Optionen
        if i % 1000 == 0:
            print(f"🔄 {i}/{len(options)} Optionen verarbeitet...")

    # Preise in Batches updaten
    update_query = f"UPDATE {table_name} SET BSM_Price = ? WHERE id = ?"
    cursor.executemany(update_query, updates)
    conn.commit()

    print(f"\n✅ {len(updates)} Optionen in {table_name} erfolgreich aktualisiert.")


def main():
    # 👉 Pfad zur Datenbank angeben:
    db_path = r"C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/data_analysis/data/sorted_option_data_db.sqlite"
    conn = sqlite3.connect(db_path)

    # Tabellen verarbeiten
    for table in ["sorted_ndx_data", "sorted_spx_data"]:
        add_bsm_price_column(conn, table)  # Spalte hinzufügen
        calculate_and_update_bsm_prices(conn, table)  # Alle Optionen berechnen und speichern

    conn.close()
    print("\n🏁 Alle Berechnungen abgeschlossen!")


if __name__ == "__main__":
    main()

