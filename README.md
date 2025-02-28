# README: Datensammlung und Analyse von Optionskontrakten

## **1. ∆∆ Einleitung**
Dieses Projekt sammelt, verarbeitet und analysiert Optionsmarktdaten mithilfe der **Polygon.io API**, **Yahoo Finance** und der **FRED API**. Die Daten werden in einer SQLite-Datenbank gespeichert und anschließend analysiert.

## **2. ∆∆ Einrichtung & Voraussetzungen**
Bevor du das Skript ausführen kannst, stelle sicher, dass folgende Komponenten installiert sind:

### **2.1 Benötigte Python-Pakete**
Installiere alle Abhängigkeiten mit folgendem Befehl:
```bash
pip install -r requirements.txt
```

### **2.2 .env Datei Konfiguration**
Die `.env` Datei enthält alle wichtigen Konfigurationsvariablen für das Projekt. Stelle sicher, dass sie im Hauptverzeichnis liegt und folgende Parameter definiert sind:

#### **API Keys & Grundlegende Parameter**
```ini
POLYGON_API_KEY=dein_polygon_api_schluessel
FRED_API_KEY=dein_fred_api_schluessel
```

#### **Datenbankspeicherorte**
```ini
RAW_DATABASE_PATH=data_collection/data
DB_ANALYSIS_PATH=data_analysis/databases
DB_ANALYSIS_FILENAME=options_analysis.sqlite
```

#### **Zeiträume & Trading Einstellungen**
```ini
POLYGON_START_DATE=2023-01-01
POLYGON_END_DATE=2023-12-31
CLOSING_CANDLES_START_DATE=2023-01-01
CLOSING_CANDLES_END_DATE=2023-12-31
INTEREST_RATE_START_DATE=2023-01-01
INTEREST_RATE_END_DATE=2023-12-31
IMPLIED_VOLATILITY_START_DATE=2023-01-01
IMPLIED_VOLATILITY_END_DATE=2023-12-31
```

#### **Indizes zur Analyse & Dynamische Datenbank-Namen**
```ini
INDICES_TO_ANALYZE=SPX,NDX
```
Basierend auf dieser Liste werden die Rohdatenbanken und Analysedateien automatisch erstellt:
- Rohdatenbank: `rawdata_spx_db.sqlite`, `rawdata_ndx_db.sqlite`
- Analysedatenbank: `sorted_spx_data`, `sorted_ndx_data`

#### **Sonstige Einstellungen**
```ini
BATCH_SIZE=10000  # Anzahl der Datensätze pro Insert-Batch
POLYGON_LIMIT=1000  # Max. Anzahl an Contracts pro API-Abfrage
POLYGON_EXPIRED=False  # Ob auch abgelaufene Kontrakte geladen werden sollen
```

#### **Steuerung der Datenverarbeitung**
Aktiviere oder deaktiviere verschiedene Verarbeitungsstufen mit diesen Schaltern:
```ini
RUN_DATA_COLLECTION=True
RUN_FETCH_CONTRACTS=True
RUN_FETCH_AGGREGATES=True
RUN_FETCH_CLOSING_CANDLES=True
RUN_FETCH_FRED_INTEREST=True
RUN_FETCH_IMPLIED_VOLATILITY=True
RUN_DATA_ANALYSIS=True
RUN_DATA_HANDLER=True
RUN_DATA_ANALYZER=True
```

---

## **3. ∆∆ Skripte & Ablauf**

### **3.1 Datensammlung (`data_collection` Bereich)**
Das Skript ruft verschiedene Marktdaten ab und speichert sie in einer SQLite-Datenbank.
- **Optionskontrakte** werden von Polygon.io geladen.
- **Aggregierte Daten** (Preisverlauf) der Optionskontrakte werden gespeichert.
- **Indexpreise** (z.B. S&P 500) werden über Yahoo Finance geladen.
- **Zinssätze** werden von FRED geladen.
- **Implizite Volatilität** wird über Yahoo Finance geholt.

**Befehl zum Starten:**
```bash
python main.py
```

### **3.2 Datenverarbeitung & Speicherung (`data_analysis` Bereich)**
Nachdem die Daten gesammelt wurden, verarbeitet `DataHandler` diese:
- Datenbanken für die Analyse werden angelegt.
- Relevante Daten werden gefiltert und bereinigt.
- Optionen werden klassifiziert in **ITM (in the money)** und **OTM (out of the money)**.

### **3.3 Datenanalyse & Visualisierung**
- `DataAnalyzer` erstellt Diagramme, um die **Verteilung von ITM/OTM-Optionen** über verschiedene Monate und Indizes hinweg zu visualisieren.
- Ergebnisse werden mit `matplotlib` dargestellt.

---

## **4. ∆∆ Troubleshooting & Fehlerbehebung**
Falls Probleme auftreten, findest du die Log-Datei unter:
```bash
error_logging/error_log.txt
```

**Typische Fehler:**
1. **SQLite Fehler: `no such table: sorted_spx_data`**
   - Mögliche Ursache: Die Migration wurde nicht ausgeführt.
   - **Lösung:** Stelle sicher, dass `RUN_DATA_HANDLER=True` gesetzt ist und starte das Skript erneut.

2. **`NoneType object is not subscriptable` (Fehler in `interest_rate`)**
   - Ursache: Keine historischen Zinssätze gefunden.
   - **Lösung:** Der Code wurde aktualisiert, sodass maximal **10 Tage zurück gesucht wird**, falls keine Zinsen gefunden wurden.

3. **API-Rate Limits erreicht (`429 Too Many Requests`)**
   - **Lösung:** Das Skript enthält eine **Retry-Logik mit exponentiellem Backoff**. Falls der Fehler anhält, reduziere die Anzahl der parallelen Requests (`max_workers` in `PolygonApiClient`).

---

## **5. ∆∆ Erweiterungen & Anpassungen**
Falls du weitere Indizes analysieren möchtest, füge sie einfach in der `.env` Datei hinzu:
```ini
INDICES_TO_ANALYZE=SPX,NDX,DAX
```
Das Skript wird automatisch die entsprechenden Datenbanken und Tabellen für diese Indizes erstellen.

Falls du nur **einen bestimmten Teil der Analyse** ausführen möchtest, kannst du die Flags in `.env` anpassen, z.B.:
```ini
RUN_DATA_COLLECTION=False
RUN_DATA_ANALYSIS=True
```
Damit wird **nur die Analyse** ausgeführt, ohne neue Daten zu laden.

---

## **6. ∆∆ Fazit**
Dieses Projekt automatisiert die Datensammlung und Analyse von Optionsmärkten.
Durch die flexible `.env` Konfiguration kannst du einfach **neue Indizes hinzufügen**, verschiedene **Datenquellen aktivieren/deaktivieren** und eine **automatische Verarbeitung** sicherstellen.

Falls du Fragen oder Probleme hast, schau in die Log-Dateien oder passe die `.env` Datei an. 🚀

