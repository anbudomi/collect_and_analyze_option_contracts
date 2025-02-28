import sqlite3
import numpy as np
import re

def add_columns_if_not_exist(conn, table_name):
    """Fügt die Spalten 'upper_bound' und 'lower_bound' hinzu, falls sie nicht existieren."""
    cursor = conn.cursor()

    # Bestehende Spalten abfragen
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor.fetchall()]

    # Spalten hinzufügen, falls nicht vorhanden
    if "upper_bound" not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN upper_bound REAL")
        print(f"✅ Spalte 'upper_bound' zu {table_name} hinzugefügt.")
    else:
        print(f"ℹ️ Spalte 'upper_bound' existiert bereits in {table_name}.")

    if "lower_bound" not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN lower_bound REAL")
        print(f"✅ Spalte 'lower_bound' zu {table_name} hinzugefügt.")
    else:
        print(f"ℹ️ Spalte 'lower_bound' existiert bereits in {table_name}.")


def calculate_bounds_and_update(conn, table_name):
    """Berechnet Wertgrenzen und schreibt sie in die Datenbank."""
    cursor = conn.cursor()

    # Daten abrufen
    query = f"""
    SELECT id, ticker, execution_price, market_price_base, remaining_time, risk_free_interest
    FROM {table_name}
    """
    cursor.execute(query)
    options = cursor.fetchall()

    print(f"\n📊 Berechne und aktualisiere Wertgrenzen für Tabelle: {table_name}")

    updates = []
    for option in options:
        option_id, ticker, K, S, T, r = option
        T_years = T / 365  # Restlaufzeit in Jahren

        # Optionstyp erkennen (robust mit Regex)
        if re.search(r'C\d{6}', ticker.upper()):  # Call
            upper_bound = S
            lower_bound = max(S - K * np.exp(-r * T_years), 0)
        elif re.search(r'P\d{6}', ticker.upper()):  # Put
            upper_bound = K * np.exp(-r * T_years)
            lower_bound = max(upper_bound - S, 0)
        else:
            print(f"⚠️ Unbekannter Optionstyp: {ticker} (ID: {option_id}) – übersprungen.")
            continue

        updates.append((upper_bound, lower_bound, option_id))

    # Werte in die Datenbank schreiben
    update_query = f"""
    UPDATE {table_name}
    SET upper_bound = ?, lower_bound = ?
    WHERE id = ?
    """
    cursor.executemany(update_query, updates)
    conn.commit()

    print(f"✅ {len(updates)} Einträge in {table_name} erfolgreich aktualisiert.")


def main():
    # 👉 Pfad zur Datenbank hier angeben:
    db_path = r"C:/Users/bulla/OneDrive/Desktop/BA_Joshua_Bullacher/data_analysis/data/sorted_option_data_db.sqlite"
    conn = sqlite3.connect(db_path)

    # Tabellen verarbeiten
    for table in ["sorted_ndx_data", "sorted_spx_data"]:
        add_columns_if_not_exist(conn, table)  # Spalten hinzufügen
        calculate_bounds_and_update(conn, table)  # Werte berechnen und updaten

    conn.close()
    print("\n🏁 Fertig!")


if __name__ == "__main__":
    main()

