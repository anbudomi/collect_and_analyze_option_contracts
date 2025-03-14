import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import calendar

# ----------------------------------------------------------------
# 1) Helfer-Funktion
# ----------------------------------------------------------------

#region 1) Helfer-Funktion
def query_db(db_path, query):
    """Führt den gegebenen SQL-Query in der Datenbank aus und gibt ein DataFrame zurück."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
#endregion


# ----------------------------------------------------------------
# 2) DataAnalyzer Klasse
# ----------------------------------------------------------------

#region 2) DataAnalyzer Klasse
class DataAnalyzer:
    def __init__(self, prepared_db_path, indices):
        self.prepared_db_path = prepared_db_path
        self.indices = indices
        self.db_connection = sqlite3.connect(self.prepared_db_path)
        self.cursor = self.db_connection.cursor()
        self.plots_dir = None  # wird in run_analysis_and_plotting gesetzt

    # ----------------------------------------------------------------
    # 2.1) Hauptmethode: run_analysis_and_plotting
    # ----------------------------------------------------------------

    #region 2.1) Hauptmethode: run_analysis_and_plotting
    def run_analysis_and_plotting(self):
        # Erstelle das Verzeichnis, falls es nicht existiert
        self.plots_dir = os.path.join("data_analysis", "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        # 2.2) Plottet die Anzahl der Optionsverträge von 2018 bis 2022
        self.plot_amount_of_options()

        # 2.3) Plottet die Anzahl der Optionsverträge nach Restlauf-Kategorie
        self.amount_of_options_per_remaining_time()

        # 2.4) Plottet die Anzahl der Optionsverträge nach Moneyness-Kategorie
        self.get_moneyness_relation()

        # 2.5) Plottet monatliche durchschnittliche relative und absolute Abweichung 2018 und 2019
        self.abs_rel_deviation_in_2018_and_2019()

        # 2.6) Plottet monatliche relative Abweichung nach Moneyness (2018-2019)
        self.moneyness_monthly_rel_error_2018_2019()

        # 2.7) Plottet die gesamte durchschnittliche relative Abweichung (2018-2019)
        self.get_total_rel_error_2018_2019()

        # 2.8) Plottet die relative Abweichung nach Moneyness (2018-2019)
        self.rel_error_corr_moneyness_2018_2019()

        # 2.9) Gibt die Anzahl der Optionen nach Moneyness-Kategorie aus (2018-2019)
        self.count_options_by_moneyness_2018_2019()
        self.debug_moneyness_counts_per_year()

        # 2.10) Berechnet die gesamte durchschnittliche absolute Abweichung (2018-2019)
        self.get_total_abs_error_2018_2019()

        # 2.11) Plottet die monatliche durchschnittliche relative Abweichung nach Restlaufzeit (2018-2019)
        self.monthly_avg_rel_error_corr_remaining_days()

        # 2.12) Plottet die durchschnittliche relative Abweichung nach Moneyness & Restlaufzeit (2018-2019)
        self.avg_rel_error_by_moneyness_remaining_days_2018_2019()

        # 2.13) Plottet die relative Abweichung für 2020: monatlich, wöchentlich und über 2020-2022
        self.monthly_rel_error_for_2020()
        self.weekly_rel_error_for_2020()
        self.monthly_rel_error_for_2020_2022()

        # 2.14) Plottet die monatliche relative Abweichung nach Moneyness und verbleibenden Tagen (2020)
        self.monthly_rel_error_moneyness_and_days_2020()

        # 2.15) Analysiert die Moneyness in Abhängigkeit von der Restlauf-Kategorie (Feb.-Apr. 2020)
        self.analyze_moneyness_by_remaining_days_feb_mar_apr_2020()

        # 2.16) Plottet die monatliche Gesamtzahl der Contracts (2020)
        self.monthly_total_contracts_2020()
    #endregion

    # ----------------------------------------------------------------
    # 2.2) Plottet die Anzahl der Optionsverträge von 2018 bis 2022
    # ----------------------------------------------------------------

    #region 2.2) Plottet die Anzahl der Optionsverträge von 2018 bis 2022
    def plot_amount_of_options(self):
        # Erstellt ein Liniendiagramm der Anzahl der Optionen pro Tag
        plt.figure(figsize=(12, 6))

        # Optionale Farbzuordnung für bekannte Ticker
        color_mapping = {"NDX": "navy", "SPX": "deepskyblue"}
        alpha_value = 0.7

        # Iteriere über alle Ticker in self.indices
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date FROM {table_name}"
            data = query_db(self.prepared_db_path, query)

            # Konvertiere 'date' in ein datetime-Format
            data['date'] = pd.to_datetime(data['date'])

            # Gruppiere nach Datum und zähle die Einträge
            count_series = data.groupby('date').size()

            # Bestimme den angepassten Labeltext für bekannte Ticker
            if ticker.upper() == "NDX":
                label_text = "Nasdaq 100"
            elif ticker.upper() == "SPX":
                label_text = "S&P 500"
            else:
                label_text = ticker

            # Ermittle die Farbe, falls definiert
            color = color_mapping.get(ticker.upper(), None)

            # Plotte die Zeitreihe
            plt.plot(count_series.index, count_series, label=label_text, color=color, alpha=alpha_value)

            # Ausgabe der Gesamtanzahl der Optionen für den Ticker
            total_count = len(data)
            print(f"Gesamtanzahl der {label_text}-Optionen: {total_count}")

        # Formatierung des Diagramms
        plt.xlabel("Zeit")
        plt.ylabel("Anzahl der Optionen")
        plt.title("Anzahl der verfügbaren Optionen über die Zeit")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # Speichere den Plot im gewünschten Verzeichnis
        save_path = os.path.join(self.plots_dir, "anzahl_options_pro_tag.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()
    #endregion

    # ----------------------------------------------------------------
    # 2.3) Plottet die Anzahl der Optionsverträge nach Restlauf-Kategorie
    # ----------------------------------------------------------------

    #region 2.3) Plottet die Anzahl der Optionsverträge nach Restlauf-Kategorie
    def amount_of_options_per_remaining_time(self):
        """
        Erstellt ein gruppiertes Balkendiagramm, das die Anzahl der Optionen pro
        Restlaufzeit-Kategorie (0-30 Tage, 31-90 Tage, >90 Tage) vergleicht.
        Der Plot wird als "plot_restlaufzeit.png" gespeichert.
        Hier wird die Spalte 'remaining_days' verwendet, die die Anzahl der verbleibenden Tage angibt.
        """
        # Definiere die Kategorien und Bins für remaining_days
        categories = ['0-30 Tage', '31-90 Tage', '>90 Tage']
        bins = [0, 30, 90, np.inf]

        # Farbzuordnung für bekannte Ticker
        color_mapping = {"NDX": "blue", "SPX": "deepskyblue"}

        # Dictionary, um die gezählten Werte pro Ticker zu speichern
        counts_dict = {}

        # Iteriere über alle Ticker in self.indices
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT remaining_days FROM {table_name}"
            data = query_db(self.prepared_db_path, query)

            # Kategorisiere die remaining_days
            data['restlaufzeit_category'] = pd.cut(data['remaining_days'], bins=bins, labels=categories,
                                                   include_lowest=True)

            # Zähle die Anzahl der Optionen je Kategorie
            counts = data['restlaufzeit_category'].value_counts().reindex(categories, fill_value=0)
            counts_dict[ticker] = counts

            # Ausgabe der Gesamtanzahl der Optionen für den Ticker
            total_count = len(data)
            # Angepasste Beschriftung
            if ticker.upper() == "NDX":
                label_text = "Nasdaq 100"
            elif ticker.upper() == "SPX":
                label_text = "S&P 500"
            else:
                label_text = ticker
            print(f"Gesamtanzahl der {label_text}-Optionen: {total_count}")

        # Erstelle das Balkendiagramm
        fig, ax = plt.subplots(figsize=(8, 6))

        x = np.arange(len(categories))
        n_tickers = len(self.indices)
        bar_width = 0.8 / n_tickers  # Verteilt die Balken in einem 80% breiten Bereich

        # Plotte die Balken für jeden Ticker
        for i, ticker in enumerate(self.indices):
            # Berechne die Position des Balkens
            positions = x - 0.4 + i * bar_width + bar_width / 2
            color = color_mapping.get(ticker.upper(), None)
            # Angepasste Beschriftung
            if ticker.upper() == "NDX":
                label_text = "Nasdaq 100"
            elif ticker.upper() == "SPX":
                label_text = "S&P 500"
            else:
                label_text = ticker
            ax.bar(positions, counts_dict[ticker], bar_width, label=label_text, color=color)

        # Achsenbeschriftungen und Formatierung
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_xlabel("Restlaufzeit-Kategorien")
        ax.set_ylabel("Anzahl der Optionen")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Formatierung der Y-Achse auf Millionen (falls gewünscht)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 1e6:.1f} Mio'))

        # Speichere den Plot im gewünschten Verzeichnis
        save_path = os.path.join(self.plots_dir, "plot_restlaufzeit.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()
    #endregion

    # ----------------------------------------------------------------
    # 2.4) Plottet die Anzahl der Optionsverträge nach Moneyness-Kategorie
    # ----------------------------------------------------------------

    #region 2.4) Plottet die Anzahl der Optionsverträge nach Moneyness-Kategorie
    def get_moneyness_relation(self, at_money_tolerance=0.01):
        """
        Erstellt ein gruppiertes Balkendiagramm, das die Verteilung der Optionen nach Moneyness zeigt.
        Negative Werte (kleiner als -at_money_tolerance) werden als "aus dem Geld",
        Werte zwischen -at_money_tolerance und at_money_tolerance als "am Geld" und
        positive Werte (größer als at_money_tolerance) als "im Geld" klassifiziert.
        Der Plot wird als "Verteilung Moneyness.png" gespeichert.
        Die Darstellung im Diagramm erfolgt in folgender Reihenfolge: "im Geld", "am Geld", "aus dem Geld".
        """
        # Definiere Bins und Original-Kategorien für die Klassifizierung
        bins = [-np.inf, -at_money_tolerance, at_money_tolerance, np.inf]
        orig_categories = ['aus dem Geld', 'am Geld', 'im Geld']
        # Definiere die gewünschte Reihenfolge der Kategorien im Plot
        plot_order = ['im Geld', 'am Geld', 'aus dem Geld']

        # Farbzuordnung für bekannte Ticker
        color_mapping = {"NDX": "blue", "SPX": "deepskyblue"}

        # Dictionary, um die gezählten Werte pro Ticker zu speichern
        counts_dict = {}

        # Iteriere über alle Ticker in self.indices
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT moneyness FROM {table_name}"
            data = query_db(self.prepared_db_path, query)

            # Kategorisiere die moneyness-Werte mit den Original-Kategorien
            data['moneyness_category'] = pd.cut(data['moneyness'], bins=bins, labels=orig_categories, include_lowest=True)

            # Zähle die Anzahl der Optionen je Kategorie und reordne nach der gewünschten Reihenfolge
            counts = data['moneyness_category'].value_counts().reindex(plot_order, fill_value=0)
            counts_dict[ticker] = counts

            total_count = len(data)
            print(f"Gesamtanzahl der {ticker}-Optionen: {total_count}")

        # Erstelle das Balkendiagramm
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(plot_order))
        n_tickers = len(self.indices)
        bar_width = 0.8 / n_tickers  # Verteilt die Balken in einem 80% breiten Bereich

        # Plotte die Balken für jeden Ticker
        for i, ticker in enumerate(self.indices):
            positions = x - 0.4 + i * bar_width + bar_width / 2
            color = color_mapping.get(ticker.upper(), None)
            label_text = "Nasdaq 100" if ticker.upper() == "NDX" else ("S&P 500" if ticker.upper() == "SPX" else ticker)
            ax.bar(positions, counts_dict[ticker], bar_width, label=label_text, color=color)

        # Achsenbeschriftungen und Formatierung
        ax.set_xticks(x)
        ax.set_xticklabels(plot_order)
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Anzahl der Optionen")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f} Mio'))

        # Speichere den Plot im gewünschten Verzeichnis
        save_path = os.path.join(self.plots_dir, "Verteilung Moneyness.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()
    #endregion

    # ----------------------------------------------------------------
    # 2.5) Plottet monatliche durchschnittliche relative und absolute Abweichung 2018 und 2019
    # ----------------------------------------------------------------

    #region 2.5) Plottet monatliche durchschnittliche relative und absolute Abweichung 2018 und 2019
    def abs_rel_deviation_in_2018_and_2019(self):
        # Dictionary zum Speichern der monatlich aggregierten Daten pro Ticker
        data_dict = {}

        # Iteriere über alle Ticker in self.indices
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            # Abfrage nur der Spalte relative_error (keine absolute_error)
            query = f"SELECT date, relative_error FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung
            df['date'] = pd.to_datetime(df['date'])
            # Filter: Nur 2018 und 2019
            df = df[(df['date'].dt.year >= 2018) & (df['date'].dt.year <= 2019)]
            # Monatliche Aggregation
            df_monthly = df.resample('ME', on='date').mean()
            # Umrechnung des relativen Fehlers in Prozent
            df_monthly['relative_error'] = df_monthly['relative_error'] * 100
            data_dict[ticker] = df_monthly

            # Ausgabe der aggregierten Ergebnisse (nur relative Abweichung)
            print(f"Monatliche durchschnittliche relative Abweichung ({ticker}):")
            print(df_monthly[['relative_error']])
            print("\n" + "-" * 50 + "\n")

        # Erstelle das Liniendiagramm für die relative Abweichung
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

        for ticker, df_monthly in data_dict.items():
            if ticker.upper() == "NDX":
                label = "Nasdaq 100"
                color = "navy"
            elif ticker.upper() == "SPX":
                label = "S&P 500"
                color = "deepskyblue"
            else:
                label = ticker
                color = None
            # Dünne, transparente Linie verbinden die Punkte
            ax.plot(df_monthly.index, df_monthly['relative_error'], label=label, color=color,
                    marker='o', linestyle='-', linewidth=0.5, alpha=0.5)
            # Punkte mit voller Deckkraft
            ax.plot(df_monthly.index, df_monthly['relative_error'], color=color,
                    marker='o', linestyle='None', alpha=1)

        ax.set_title("Monatliche durchschnittliche relative Abweichung (2018-2019)")
        ax.set_ylabel("Relative Abweichung (%)")
        ax.set_xlabel("Zeit")  # Hier wird "Zeit" als x-Achsen-Beschriftung hinzugefügt.
        leg = ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Anpassung der Legenden: Dickere, weniger transparente Linien
        for handle in leg.legend_handles:
            try:
                handle.set_linewidth(2.0)
                handle.set_alpha(1.0)
            except Exception:
                pass

        # X-Achse formatieren: Jahresmarkierungen
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Speichere den Plot im gewünschten Verzeichnis
        save_path = os.path.join(self.plots_dir, "relative_deviation_2018_2019.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()

    #endregion

    # ----------------------------------------------------------------
    # 2.6) Plottet monatliche relative Abweichung nach Moneyness (2018-2019)
    # ----------------------------------------------------------------

    #region 2.6) Plottet monatliche relative Abweichung nach Moneyness (2018-2019)
    def moneyness_monthly_rel_error_2018_2019(self, at_money_tolerance=0.01):
        """
        Erstellt ein Diagramm mit zwei Subplots, das die monatliche durchschnittliche
        relative Abweichung (in %) für jede Moneyness-Kategorie ("im Geld", "am Geld",
        "aus dem Geld") für S&P 500 und Nasdaq 100 im Zeitraum 2018-2019 zeigt.
        Die Daten werden als Punkte dargestellt, die durch eine dünne, transparente Linie
        verbunden sind, sodass der Zusammenhang der monatlichen Durchschnittswerte erkennbar bleibt.
        Der Plot wird als "Monatliche relative Abweichung Moneyness in 2018_2019.png"
        im Plot-Verzeichnis gespeichert.

        Dabei wird die Spalte "moneyness" aus der Datenbank genutzt. Werte > at_money_tolerance
        gelten als "im Geld", Werte < -at_money_tolerance als "aus dem Geld" und Werte innerhalb
        von ±at_money_tolerance als "am Geld". Die relative Abweichung wird in Prozent angegeben.
        """
        # Dictionary zum Speichern der monatlich aggregierten Daten pro Ticker
        monthly_dict = {}

        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, relative_error, moneyness, ticker FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung für 2018 und 2019
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.year >= 2018) & (df['date'].dt.year <= 2019)]

            # Moneyness-Kategorien definieren:
            # Werte < -at_money_tolerance -> "aus dem Geld"
            # -at_money_tolerance bis at_money_tolerance -> "am Geld"
            # Werte > at_money_tolerance -> "im Geld"
            bins = [-np.inf, -at_money_tolerance, at_money_tolerance, np.inf]
            labels = ['aus dem Geld', 'am Geld', 'im Geld']
            df['moneyness_category'] = pd.cut(df['moneyness'], bins=bins, labels=labels, include_lowest=True)

            # Monatliche Aggregation: Gruppierung nach Moneyness-Kategorie und Monat,
            # Mittelwert der relativen Abweichung berechnen und in Prozent umrechnen
            df_monthly = df.groupby(
                ['moneyness_category', pd.Grouper(key='date', freq='ME')],
                observed=False
            )['relative_error'].mean().unstack(0)
            df_monthly = df_monthly * 100  # Umrechnung in Prozent
            monthly_dict[ticker] = df_monthly

            print(f"Monatliche durchschnittliche relative Abweichung ({ticker}):")
            print(df_monthly)
            print("\n" + "-" * 50 + "\n")

        # Erstelle das Diagramm mit zwei Subplots (oben: SPX, unten: NDX)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

        # Plot für S&P 500
        for t in self.indices:
            if t.upper() == "SPX":
                spx_monthly = monthly_dict[t]
                axes[0].plot(spx_monthly.index, spx_monthly['im Geld'], label='SPX - im Geld', color='blue',
                             marker='o', linestyle='-', linewidth=0.5, alpha=0.5)
                axes[0].plot(spx_monthly.index, spx_monthly['am Geld'], label='SPX - am Geld', color='green',
                             marker='o', linestyle='-', linewidth=0.5, alpha=0.5)
                axes[0].plot(spx_monthly.index, spx_monthly['aus dem Geld'], label='SPX - aus dem Geld', color='red',
                             marker='o', linestyle='-', linewidth=0.5, alpha=0.5)
                # Punkte mit voller Deckkraft
                axes[0].plot(spx_monthly.index, spx_monthly['im Geld'], color='blue', marker='o', linestyle='None',
                             alpha=1)
                axes[0].plot(spx_monthly.index, spx_monthly['am Geld'], color='green', marker='o', linestyle='None',
                             alpha=1)
                axes[0].plot(spx_monthly.index, spx_monthly['aus dem Geld'], color='red', marker='o', linestyle='None',
                             alpha=1)
                axes[0].set_title("Monatliche relative Abweichung für S&P 500 (2018-2019)")
                axes[0].set_ylabel("Relative Abweichung (%)")
                leg0 = axes[0].legend()
                axes[0].grid(True, linestyle='--', alpha=0.7)

        # Plot für Nasdaq 100
        for t in self.indices:
            if t.upper() == "NDX":
                ndx_monthly = monthly_dict[t]
                axes[1].plot(ndx_monthly.index, ndx_monthly['im Geld'], label='NDX - im Geld', color='blue',
                             marker='o', linestyle='-', linewidth=0.5, alpha=0.5)
                axes[1].plot(ndx_monthly.index, ndx_monthly['am Geld'], label='NDX - am Geld', color='green',
                             marker='o', linestyle='-', linewidth=0.5, alpha=0.5)
                axes[1].plot(ndx_monthly.index, ndx_monthly['aus dem Geld'], label='NDX - aus dem Geld', color='red',
                             marker='o', linestyle='-', linewidth=0.5, alpha=0.5)
                # Punkte mit voller Deckkraft
                axes[1].plot(ndx_monthly.index, ndx_monthly['im Geld'], color='blue', marker='o', linestyle='None',
                             alpha=1)
                axes[1].plot(ndx_monthly.index, ndx_monthly['am Geld'], color='green', marker='o', linestyle='None',
                             alpha=1)
                axes[1].plot(ndx_monthly.index, ndx_monthly['aus dem Geld'], color='red', marker='o', linestyle='None',
                             alpha=1)
                axes[1].set_title("Monatliche relative Abweichung für Nasdaq 100 (2018-2019)")
                axes[1].set_ylabel("Relative Abweichung (%)")
                axes[1].set_xlabel("Zeit")
                leg1 = axes[1].legend()
                axes[1].grid(True, linestyle='--', alpha=0.7)

        # Anpassung der Legenden: Dickere, weniger transparente Linien
        for leg in [leg0, leg1]:
            for handle in leg.legend_handles:
                try:
                    handle.set_linewidth(2.0)
                    handle.set_alpha(1.0)
                except Exception:
                    pass

        # X-Achse formatieren: Jahresmarkierungen
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Speichere den Plot im gewünschten Verzeichnis
        save_path = os.path.join(self.plots_dir, "Monatliche relative Abweichung Moneyness in 2018_2019.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()
    #endregion

    # ----------------------------------------------------------------
    # 2.7) Plottet die gesamte durchschnittliche relative Abweichung (2018-2019)
    # ----------------------------------------------------------------

    #region 2.7) Plottet die gesamte durchschnittliche relative Abweichung (2018-2019)
    def get_total_rel_error_2018_2019(self, at_money_tolerance=0.01):
        """
        Berechnet für Nasdaq 100 und S&P 500 über den Zeitraum 2018-2019
        die gesamte durchschnittliche relative Abweichung (in %) getrennt nach den
        Moneyness-Kategorien:
          - "im Geld"  : moneyness > at_money_tolerance
          - "am Geld"  : -at_money_tolerance <= moneyness <= at_money_tolerance
          - "aus dem Geld": moneyness < -at_money_tolerance
        Es wird ausschließlich der durchschnittliche relative Fehler (in %) pro
        Kategorie ausgegeben.
        """
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, relative_error, moneyness FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung für 2018 und 2019
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.year >= 2018) & (df['date'].dt.year <= 2019)]

            # Umrechnung der relativen Abweichung in Prozent
            df['relative_error_percent'] = df['relative_error'] * 100

            # Moneyness-Kategorien definieren:
            # Werte < -at_money_tolerance -> "aus dem Geld"
            # -at_money_tolerance bis at_money_tolerance -> "am Geld"
            # Werte > at_money_tolerance -> "im Geld"
            bins = [-np.inf, -at_money_tolerance, at_money_tolerance, np.inf]
            labels = ['aus dem Geld', 'am Geld', 'im Geld']
            df['moneyness_category'] = pd.cut(df['moneyness'], bins=bins, labels=labels, include_lowest=True)

            # Gruppiere nach Moneyness-Kategorie und berechne den Mittelwert des relativen Fehlers
            grouped = df.groupby('moneyness_category')['relative_error_percent'].mean()

            # Angepasste Ticker-Beschriftung
            if ticker.upper() == "NDX":
                label_text = "Nasdaq 100"
            elif ticker.upper() == "SPX":
                label_text = "S&P 500"
            else:
                label_text = ticker

            # Ausgabe der Ergebnisse
            print(f"Gesamte durchschnittliche relative Abweichung ({label_text}, 2018-2019):")
            print(grouped)
            print("-" * 50)
    #endregion

    # ----------------------------------------------------------------
    # 2.8) Plottet die relative Abweichung nach Moneyness (2018-2019)
    # ----------------------------------------------------------------

    #region 2.8) Plottet die relative Abweichung nach Moneyness (2018-2019)
    def rel_error_corr_moneyness_2018_2019(self, at_money_tolerance=0.01):
        """
        Berechnet für Nasdaq 100 und S&P 500 im Zeitraum 2018-2019 die durchschnittliche
        relative Abweichung (in %) für aus dem Geld liegende Optionen, abhängig davon, wie weit
        sie aus dem Geld sind. Es werden alle Optionen betrachtet, bei denen moneyness < -at_money_tolerance
        (also < -0.01) gilt. Die "excess moneyness" wird in folgende Kategorien eingeteilt:
          - Bin 1: "< -0.06"
          - Bin 2: "-0.06 to < -0.05"
          - Bin 3: "-0.05 to < -0.04"
          - Bin 4: "-0.04 to < -0.03"
          - Bin 5: "-0.03 to < -0.02"
          - Bin 6: "-0.02 to < -0.01"
        Da in der Datenbank nur die Spalte "relative_error" vorliegt, wird diese in Prozent umgerechnet.
        Die Ergebnisse werden in zwei Balkendiagrammen (oben: S&P 500, unten: Nasdaq 100) dargestellt
        und der Plot wird als "Durchschnittliche rel Abweichung nach Moneyness 2019_2019.png"
        im Plot-Verzeichnis gespeichert.
        (Die Reihenfolge der x-Achsen-Beschriftung wird umgedreht.)
        """
        group_dict = {}

        # Definiere Bins und Labels für die aus-dem-Geld Kategorisierung (im negativen Bereich)
        bins = [-np.inf, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01]
        labels = ["< -0.06", "-0.06 to < -0.05", "-0.05 to < -0.04", "-0.04 to < -0.03", "-0.03 to < -0.02",
                  "-0.02 to < -0.01"]

        # Um die Reihenfolge der x-Achse umzukehren, definieren wir die gewünschte Reihenfolge als Umkehrung
        rev_labels = labels[::-1]

        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, relative_error, moneyness, ticker FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung für 2018 und 2019
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.year >= 2018) & (df['date'].dt.year <= 2019)]

            # Betrachte nur aus dem Geld liegende Optionen: moneyness < -at_money_tolerance
            df_otm = df[df['moneyness'] < -at_money_tolerance].copy()

            # Umrechnung des relativen Fehlers in Prozent
            df_otm['relative_error_percent'] = df_otm['relative_error'] * 100

            # Kategorisierung der moneyness in den definierten negativen Intervallen
            df_otm['moneyness_bin'] = pd.cut(df_otm['moneyness'], bins=bins, labels=labels, include_lowest=True)

            # Berechne den durchschnittlichen relativen Fehler pro Moneyness-Bin
            grouped = df_otm.groupby('moneyness_bin')['relative_error_percent'].mean()
            # Reindexiere, um die Reihenfolge der x-Achsen-Beschriftungen umzukehren
            grouped = grouped.reindex(rev_labels)
            group_dict[ticker] = grouped

            print(f"Durchschnittliche relative Abweichung nach Moneyness ({ticker}, 2018-2019):")
            print(grouped)
            print("-" * 50)

        # Erstelle das Balkendiagramm in zwei Subplots
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

        # Plot für S&P 500
        for t in self.indices:
            if t.upper() == "SPX":
                spx_grouped = group_dict[t]
                axes[0].bar(spx_grouped.index.astype(str), spx_grouped, color='deepskyblue')
                axes[0].set_title("Durchschnittliche relative Abweichung nach Moneyness (S&P 500, 2018-2019)")
                axes[0].set_ylabel("Relative Abweichung (%)")
                axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Plot für Nasdaq 100
        for t in self.indices:
            if t.upper() == "NDX":
                ndx_grouped = group_dict[t]
                axes[1].bar(ndx_grouped.index.astype(str), ndx_grouped, color='navy')
                axes[1].set_title("Durchschnittliche relative Abweichung nach Moneyness (Nasdaq 100, 2018-2019)")
                axes[1].set_xlabel("Moneyness-Kategorien")
                axes[1].set_ylabel("Relative Abweichung (%)")
                axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Speichere den Plot im gewünschten Verzeichnis
        save_path = os.path.join(self.plots_dir, "Durchschnittliche rel Abweichung nach Moneyness 2019_2019.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()
    #endregion

    # ----------------------------------------------------------------
    # 2.9) Gibt die Anzahl der Optionen nach Moneyness-Kategorie aus (2018-2019)
    # ----------------------------------------------------------------

    #region 2.9) Gibt die Anzahl der Optionen nach Moneyness-Kategorie aus (2018-2019)
    def count_options_by_moneyness_2018_2019(self, at_money_tolerance=0.01):
        """
        Berechnet für Nasdaq 100 und S&P 500 im Zeitraum 2018-2019 die Anzahl der Optionen,
        getrennt nach Moneyness-Kategorien, wobei nur aus dem Geld liegende Optionen betrachtet werden
        (d.h. solche mit moneyness < -at_money_tolerance, also < -0.01).
        Die "excess moneyness" wird in folgende Intervalle unterteilt:
          - Bin 1: "< -0.06"
          - Bin 2: "-0.06 to < -0.05"
          - Bin 3: "-0.05 to < -0.04"
          - Bin 4: "-0.04 to < -0.03"
          - Bin 5: "-0.03 to < -0.02"
          - Bin 6: "-0.02 to < -0.01"
        Die Ergebnisse werden für jeden Ticker formatiert ausgegeben.
        """
        count_dict = {}
        bins = [-np.inf, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01]
        labels = ["< -0.06", "-0.06 to < -0.05", "-0.05 to < -0.04", "-0.04 to < -0.03", "-0.03 to < -0.02", "-0.02 to < -0.01"]

        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, moneyness, ticker FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung für 2018 und 2019
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.year >= 2018) & (df['date'].dt.year <= 2019)]

            # Betrachte nur aus dem Geld liegende Optionen: moneyness < -at_money_tolerance
            df_otm = df[df['moneyness'] < -at_money_tolerance].copy()

            # Kategorisiere die moneyness in die definierten negativen Intervalle
            df_otm['moneyness_bin'] = pd.cut(df_otm['moneyness'], bins=bins, labels=labels, right=False, include_lowest=True)

            # Zähle die Anzahl der Optionen pro Kategorie und reindexiere, um alle Labels anzuzeigen
            counts = df_otm['moneyness_bin'].value_counts().reindex(labels, fill_value=0)
            count_dict[ticker] = counts

            # Angepasste Ticker-Beschriftung
            if ticker.upper() == "NDX":
                label_text = "Nasdaq 100"
            elif ticker.upper() == "SPX":
                label_text = "S&P 500"
            else:
                label_text = ticker

            # Ausgabe der Ergebnisse in einer formatierten Tabelle
            print(f"\nAnzahl der Optionen nach Moneyness-Kategorien ({label_text}, 2018-2019):")
            for cat, cnt in counts.items():
                print(f"  {cat}: {cnt}")
            print("-" * 50)

    def debug_moneyness_counts_per_year(self, at_money_tolerance=0.01):
        """
        Für jeden Ticker (z. B. Nasdaq 100 und S&P 500) im Zeitraum 2018-2022
        wird die Anzahl der Optionen pro Jahr in drei Moneyness-Kategorien ausgegeben:
          - "aus dem Geld": moneyness < -at_money_tolerance  (also Werte < -0.01)
          - "am Geld":     -at_money_tolerance <= moneyness < at_money_tolerance  (also Werte von -0.01 bis < 0.01)
          - "im Geld":     moneyness >= at_money_tolerance  (also Werte >= 0.01)
        Die Ergebnisse werden pro Jahr formatiert ausgegeben.
        """
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, moneyness, ticker FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung für 2018 bis 2022
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.year >= 2018) & (df['date'].dt.year <= 2022)]

            # Alle Optionen werden in drei Kategorien eingeteilt, wobei wir right=False verwenden,
            # damit die Intervalle links-geschlossen, rechts-offen sind:
            # "aus dem Geld": [-∞, -0.01)
            # "am Geld":     [-0.01, 0.01)
            # "im Geld":     [0.01, ∞)
            bins = [-np.inf, -at_money_tolerance, at_money_tolerance, np.inf]
            labels = ["aus dem Geld", "am Geld", "im Geld"]
            df['moneyness_category'] = pd.cut(df['moneyness'], bins=bins, labels=labels, right=False, include_lowest=True)

            # Extrahiere das Jahr
            df['year'] = df['date'].dt.year

            # Gruppiere nach Jahr und Moneyness-Kategorie und zähle die Optionen
            counts = df.groupby(['year', 'moneyness_category']).size().unstack(fill_value=0)

            # Angepasste Ticker-Beschriftung
            if ticker.upper() == "NDX":
                label_text = "Nasdaq 100"
            elif ticker.upper() == "SPX":
                label_text = "S&P 500"
            else:
                label_text = ticker

            print(f"\nMoneyness-Kategorien für {label_text} pro Jahr (2018-2022):")
            print(counts)
            print("-" * 50)
    #endregion

    # ----------------------------------------------------------------
    # 2.10) Berechnet die gesamte durchschnittliche absolute Abweichung (2018-2019)
    # ----------------------------------------------------------------

    #region 2.10) Berechnet die gesamte durchschnittliche absolute Abweichung (2018-2019)
    def get_total_abs_error_2018_2019(self, at_money_tolerance=0.01):
        """
        Berechnet für Nasdaq 100 und S&P 500 über den Zeitraum 2018-2019
        die gesamte durchschnittliche absolute Abweichung, getrennt nach den
        Moneyness-Kategorien:
          - "im Geld"  : moneyness > at_money_tolerance
          - "am Geld"  : -at_money_tolerance <= moneyness <= at_money_tolerance
          - "aus dem Geld": moneyness < -at_money_tolerance
        Es wird ausschließlich der durchschnittliche absolute Fehler pro
        Kategorie ausgegeben.
        """
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            # Wir fragen hier absolute_error statt relative_error ab.
            query = f"SELECT date, absolute_error, moneyness FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung für 2018 und 2019
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.year >= 2018) & (df['date'].dt.year <= 2019)]

            # Verwende absolute_error direkt (keine Umrechnung erforderlich)
            # Moneyness-Kategorien definieren:
            # "aus dem Geld": moneyness < -at_money_tolerance
            # "am Geld": -at_money_tolerance <= moneyness <= at_money_tolerance
            # "im Geld": moneyness > at_money_tolerance
            bins = [-np.inf, -at_money_tolerance, at_money_tolerance, np.inf]
            labels = ['aus dem Geld', 'am Geld', 'im Geld']
            df['moneyness_category'] = pd.cut(df['moneyness'], bins=bins, labels=labels, include_lowest=True)

            # Gruppiere nach Moneyness-Kategorie und berechne den Mittelwert des absoluten Fehlers
            grouped = df.groupby('moneyness_category')['absolute_error'].mean()

            # Angepasste Ticker-Beschriftung
            if ticker.upper() == "NDX":
                label_text = "Nasdaq 100"
            elif ticker.upper() == "SPX":
                label_text = "S&P 500"
            else:
                label_text = ticker

            print(f"Gesamte durchschnittliche absolute Abweichung ({label_text}, 2018-2019):")
            print(grouped)
            print("-" * 50)
    #endregion

    # ----------------------------------------------------------------
    # 2.11) Plottet die monatliche durchschnittliche relative Abweichung nach Restlaufzeit (2018-2019)
    # ----------------------------------------------------------------

    #region 2.11) Plottet die monatliche durchschnittliche relative Abweichung nach Restlaufzeit (2018-2019)
    def monthly_avg_rel_error_corr_remaining_days(self):
        """
        Erstellt ein Diagramm mit zwei Subplots, das die monatliche durchschnittliche
        relative Abweichung (in %) für jede Restlaufzeit-Kategorie für die Jahre 2018-2019 zeigt.
        Es werden drei Kategorien gebildet:
          - "0-30 Tage": Optionen mit remaining_days zwischen 0 und 30,
          - "31-90 Tage": Optionen mit remaining_days zwischen 31 und 90,
          - ">90 Tage": Optionen mit remaining_days über 90.
        Die relative Abweichung wird aus der Spalte 'relative_error' berechnet und in Prozent
        umgerechnet (relative_error * 100). Die Daten werden monatlich aggregiert.
        Der Plot wird in zwei Subplots dargestellt (oben: S&P 500, unten: Nasdaq 100)
        und als "Monatliche relative Abweichung nach Laufzeitgruppen 2018_2019.png" gespeichert.
        """
        monthly_dict = {}

        # Definiere die Bins und Labels für die Laufzeit-Gruppierung
        bins = [0, 30, 90, np.inf]
        labels = ["0-30 Tage", "31-90 Tage", ">90 Tage"]

        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, relative_error, remaining_days, ticker FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung für 2018 und 2019
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.year >= 2018) & (df['date'].dt.year <= 2019)]

            # Umrechnung des relativen Fehlers in Prozent
            df['relative_error_percent'] = df['relative_error'] * 100

            # Kategorisiere die Optionen anhand ihrer verbleibenden Laufzeit (remaining_days)
            df['remaining_days_category'] = pd.cut(df['remaining_days'], bins=bins, labels=labels, include_lowest=True)

            # Monatliche Aggregation: Gruppierung nach Laufzeit-Kategorie und Monat,
            # Berechnung des Mittelwerts der relativen Abweichung in Prozent
            df_monthly = df.groupby(['remaining_days_category', pd.Grouper(key='date', freq='ME')])['relative_error_percent'].mean().unstack(0)
            monthly_dict[ticker] = df_monthly

            print(f"Monatliche durchschnittliche relative Abweichung nach Laufzeitgruppen ({ticker}):")
            print(df_monthly)
            print("\n" + "-" * 50 + "\n")

        # Erstelle das Diagramm mit zwei Subplots (oben: SPX, unten: NDX)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

        # Farben für die Laufzeit-Gruppen
        color_mapping = {"0-30 Tage": "blue", "31-90 Tage": "green", ">90 Tage": "red"}

        # Plot für S&P 500
        for t in self.indices:
            if t.upper() == "SPX":
                spx_monthly = monthly_dict[t]
                for group in spx_monthly.columns:
                    # Dünne, transparente Linie
                    axes[0].plot(spx_monthly.index, spx_monthly[group], label=f"SPX - {group}",
                                 color=color_mapping.get(group, None), marker='o', linestyle='-', linewidth=0.5, alpha=0.5)
                    # Punkte mit voller Deckkraft
                    axes[0].plot(spx_monthly.index, spx_monthly[group], color=color_mapping.get(group, None),
                                 marker='o', linestyle='None', alpha=1)
                axes[0].set_title("Monatliche relative Abweichung nach Laufzeitgruppen (S&P 500, 2018-2019)")
                axes[0].set_ylabel("Relative Abweichung (%)")
                axes[0].legend()
                axes[0].grid(True, linestyle='--', alpha=0.7)

        # Plot für Nasdaq 100
        for t in self.indices:
            if t.upper() == "NDX":
                ndx_monthly = monthly_dict[t]
                for group in ndx_monthly.columns:
                    axes[1].plot(ndx_monthly.index, ndx_monthly[group], label=f"NDX - {group}",
                                 color=color_mapping.get(group, None), marker='o', linestyle='-', linewidth=0.5, alpha=0.5)
                    axes[1].plot(ndx_monthly.index, ndx_monthly[group], color=color_mapping.get(group, None),
                                 marker='o', linestyle='None', alpha=1)
                axes[1].set_title("Monatliche relative Abweichung nach Laufzeitgruppen (Nasdaq 100, 2018-2019)")
                axes[1].set_ylabel("Relative Abweichung (%)")
                axes[1].set_xlabel("Zeit")
                axes[1].legend()
                axes[1].grid(True, linestyle='--', alpha=0.7)

        # X-Achse formatieren: Jahresmarkierungen
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Speichere den Plot im gewünschten Verzeichnis
        save_path = os.path.join(self.plots_dir, "Monatliche relative Abweichung nach Laufzeitgruppen 2018_2019.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()
    #endregion

    # ----------------------------------------------------------------
    # 2.12) Plottet die durchschnittliche relative Abweichung nach Moneyness & Restlaufzeit (2018-2019)
    # ----------------------------------------------------------------

    #region 2.12) Plottet die durchschnittliche relative Abweichung nach Moneyness & Restlaufzeit (2018-2019)
    def avg_rel_error_by_moneyness_remaining_days_2018_2019(self, at_money_tolerance=0.01):
        """
        Erstellt zwei gruppierte Balkendiagramme (oben: Nasdaq 100, unten: S&P 500),
        die die durchschnittliche relative Abweichung (in %) nach Moneyness-Kategorie
        und Restlaufzeit-Kategorie (0-30 Tage, 31-90 Tage, >90 Tage) im Zeitraum
        2018-2019 zeigen.

        Die Moneyness-Kategorien lauten in folgender Reihenfolge:
            1) "im Geld"     (moneyness > at_money_tolerance)
            2) "am Geld"     (-at_money_tolerance <= moneyness <= at_money_tolerance)
            3) "aus dem Geld" (moneyness < -at_money_tolerance)

        Die Restlaufzeit-Kategorien lauten:
            - "0-30 Tage"   (0 <= remaining_days <= 30)
            - "31-90 Tage"  (30 < remaining_days <= 90)
            - ">90 Tage"    (remaining_days > 90)

        Die relative Abweichung stammt aus der Spalte 'relative_error' und wird
        in Prozent umgerechnet. Die Ergebnisse werden in einem gruppierten Balkendiagramm
        pro Ticker dargestellt. Der Plot wird als
        "Monatliche relative Abweichung Moneyness_Laufzeit 2018_2019.png"
        gespeichert.
        """
        # Moneyness-Bins und -Labels (reihenfolge = im Geld, am Geld, aus dem Geld)
        moneyness_bins = [-np.inf, -at_money_tolerance, at_money_tolerance, np.inf]
        moneyness_labels = ["aus dem Geld", "am Geld", "im Geld"]
        # Wir wollen "im Geld" vorne, dann "am Geld", dann "aus dem Geld" => wir reindexen später.

        # Restlaufzeit-Bins und -Labels
        days_bins = [0, 30, 90, np.inf]
        days_labels = ["0-30 Tage", "31-90 Tage", ">90 Tage"]

        # Farben für die drei Restlaufzeit-Kategorien
        colors = ["navy", "deepskyblue", "dodgerblue"]

        # Gewünschte finale Reihenfolge für die Moneyness-Achse
        final_moneyness_order = ["im Geld", "am Geld", "aus dem Geld"]

        data_dict = {}

        # --- Daten für jeden Ticker laden und verarbeiten ---
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, relative_error, moneyness, remaining_days FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Filter 2018-2019 und relative_error -> Prozent
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.year >= 2018) & (df['date'].dt.year <= 2019)].copy()
            df['relative_error_percent'] = df['relative_error'] * 100

            # Moneyness-Kategorisierung
            df['moneyness_cat'] = pd.cut(
                df['moneyness'],
                bins=moneyness_bins,
                labels=moneyness_labels,
                include_lowest=True
            )
            # Restlaufzeit-Kategorisierung
            df['days_cat'] = pd.cut(
                df['remaining_days'],
                bins=days_bins,
                labels=days_labels,
                include_lowest=True
            )

            # Gruppierung: durchschnittliche relative Abweichung pro (moneyness_cat, days_cat)
            grouped = df.groupby(['moneyness_cat', 'days_cat'])['relative_error_percent'].mean().unstack('days_cat')

            # Reindex für die gewünschte Spalten- und Zeilen-Reihenfolge
            # 1) days_labels (Spalten) in der Reihenfolge 0-30, 31-90, >90
            grouped = grouped.reindex(columns=days_labels)
            # 2) moneyness in der Reihenfolge "im Geld", "am Geld", "aus dem Geld"
            grouped = grouped.reindex(index=final_moneyness_order)

            data_dict[ticker] = grouped

            print(f"\nDurchschnittliche rel. Abweichung nach Moneyness & Laufzeit (2018-2019) – {ticker}")
            print(grouped)
            print("-" * 60)

        # --- Plot-Vorbereitung ---
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

        # --- Plot für Nasdaq 100 (oberer Subplot) ---
        for t in self.indices:
            if t.upper() == "NDX":
                ndx_grouped = data_dict[t]
                # Erstellen eines gruppierten Balkendiagramms
                ndx_plot = ndx_grouped.plot(
                    kind="bar",
                    ax=axes[0],
                    color=colors,
                    width=0.8  # Breite der Balken
                )
                axes[0].set_title("Durchschnittliche relative Abweichung - Nasdaq 100 (2018-2019)")
                axes[0].set_ylabel("Durchschnittliche relative Abweichung (%)")
                axes[0].legend(title="Laufzeitgruppe")
                axes[0].grid(axis='y', linestyle='--', alpha=0.7)
                # Werte über die Balken schreiben
                for container in axes[0].containers:
                    axes[0].bar_label(container, fmt='%.2f', padding=3)

        # --- Plot für S&P 500 (unterer Subplot) ---
        for t in self.indices:
            if t.upper() == "SPX":
                spx_grouped = data_dict[t]
                spx_plot = spx_grouped.plot(
                    kind="bar",
                    ax=axes[1],
                    color=colors,
                    width=0.8
                )
                axes[1].set_title("Durchschnittliche relative Abweichung - S&P 500 (2018-2019)")
                axes[1].set_xlabel("Moneyness-Kategorie")
                axes[1].set_ylabel("Durchschnittliche relative Abweichung (%)")
                axes[1].legend(title="Laufzeitgruppe")
                axes[1].grid(axis='y', linestyle='--', alpha=0.7)
                # Werte über die Balken schreiben
                for container in axes[1].containers:
                    axes[1].bar_label(container, fmt='%.2f', padding=3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Plot speichern
        save_path = os.path.join(self.plots_dir, "Monatliche relative Abweichung Moneyness_Laufzeit 2018_2019.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()
    #endregion

    # ----------------------------------------------------------------
    # 2.13) Plottet die relative Abweichung für 2020: monatlich, wöchentlich und über 2020-2022
    # ----------------------------------------------------------------

    #region 2.13) Plottet die relative Abweichung für 2020: monatlich, wöchentlich und über 2020-2022
    def monthly_rel_error_for_2020(self):
        """
        Erstellt ein Diagramm mit zwei Subplots (oben: S&P 500, unten: Nasdaq 100),
        das die monatliche durchschnittliche relative Abweichung (in %) für das Jahr 2020 zeigt.

        Dabei wird:
          - 'relative_error' in Prozent umgerechnet.
          - Nach Monat gruppiert und der Durchschnitt berechnet.
          - Ein Balkendiagramm (x=Monat, y=Abweichung in %) erstellt.
          - Für S&P 500 wird die Farbe 'deepskyblue' genutzt, für Nasdaq 100 'navy'.
          - Die Balken werden beschriftet mit ihrem Wert.
          - Die Monatsachse wird mit Kurzbezeichnungen (Jan, Feb, etc.) beschriftet.

        Der Plot wird als "monthly_rel_error_2020.png" im Plot-Verzeichnis gespeichert.
        """
        # Dictionary zum Speichern der monatlich aggregierten Daten
        monthly_data = {}

        # Wir fokussieren uns auf SPX (S&P 500) und NDX (Nasdaq 100).
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, relative_error FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung für das Jahr 2020
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'].dt.year == 2020].copy()

            # Umrechnung der relativen Abweichung in Prozent
            df['relative_error_percent'] = df['relative_error'] * 100

            # Monatliche Aggregation (Mittelwert)
            df['month'] = df['date'].dt.month
            grouped = df.groupby('month')['relative_error_percent'].mean()

            # Reindex auf alle 12 Monate, damit ggf. fehlende Monate als 0 erscheinen
            grouped = grouped.reindex(range(1, 13), fill_value=0)
            monthly_data[ticker] = grouped

        # Erstelle die Subplots für SPX (oben) und NDX (unten)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

        # Liste der Monatsnamen für die X-Achse
        month_names = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                       "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

        # Balkenbreite
        bar_width = 0.6

        # --- OBERES DIAGRAMM: S&P 500 (SPX) ---
        for t in self.indices:
            if t.upper() == "SPX":
                spx_2020 = monthly_data.get(t)
                x_positions = np.arange(1, 13)
                bars_spx = axes[0].bar(
                    x_positions,
                    spx_2020.values,
                    color='deepskyblue',
                    label='S&P 500',
                    width=bar_width
                )
                axes[0].set_title("Monatliche durchschnittliche relative Abweichung für S&P 500 im Jahr 2020")
                axes[0].set_ylabel("Relative Abweichung (%)")
                axes[0].legend()
                axes[0].grid(True, linestyle='--', alpha=0.7)

                # Beschriftung der Balken
                for bar in bars_spx:
                    height = bar.get_height()
                    if height != 0:  # nur beschriften, wenn Wert != 0
                        axes[0].text(
                            bar.get_x() + bar.get_width() / 2,
                            height,
                            f'{height:.2f}',
                            ha='center',
                            va='bottom',
                            fontsize=9
                        )

        # --- UNTERES DIAGRAMM: Nasdaq 100 (NDX) ---
        for t in self.indices:
            if t.upper() == "NDX":
                ndx_2020 = monthly_data.get(t)
                x_positions = np.arange(1, 13)
                bars_ndx = axes[1].bar(
                    x_positions,
                    ndx_2020.values,
                    color='navy',
                    label='Nasdaq 100',
                    width=bar_width
                )
                axes[1].set_title("Monatliche durchschnittliche relative Abweichung für Nasdaq 100 im Jahr 2020")
                axes[1].set_ylabel("Relative Abweichung (%)")
                axes[1].set_xlabel("Monate")
                axes[1].legend()
                axes[1].grid(True, linestyle='--', alpha=0.7)

                # Beschriftung der Balken
                for bar in bars_ndx:
                    height = bar.get_height()
                    if height != 0:
                        axes[1].text(
                            bar.get_x() + bar.get_width() / 2,
                            height,
                            f'{height:.2f}',
                            ha='center',
                            va='bottom',
                            fontsize=9
                        )

        # X-Achsen-Formatierung
        for ax in axes:
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names, rotation=0)

        plt.tight_layout()

        # Speichere den Plot im gewünschten Verzeichnis
        save_path = os.path.join(self.plots_dir, "monthly_rel_error_2020.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()

    def weekly_rel_error_for_2020(self):
        """
        Erstellt ein Diagramm mit zwei Subplots (oben: S&P 500, unten: Nasdaq 100),
        das die wöchentliche durchschnittliche relative Abweichung (in %) für das Jahr 2020 zeigt.

        Schritte:
          1) 'relative_error' wird in Prozent umgerechnet.
          2) Gruppierung nach Kalenderwoche (1..53) und Mittelwert-Bildung.
          3) Balkendiagramm (x=Kalenderwoche, y=Abweichung in %) pro Ticker.
          4) Keine Beschriftung der Balkenwerte, um das Diagramm übersichtlich zu halten.
          5) Zur besseren Übersicht werden die X-Ticks nur für jede 4. Woche gesetzt.

        Der Plot wird als "weekly_rel_error_2020.png" im Plot-Verzeichnis gespeichert.
        """
        # Dictionary zum Speichern der wöchentlich aggregierten Daten
        weekly_data = {}

        # Fokussieren uns auf SPX (S&P 500) und NDX (Nasdaq 100),
        # können aber auch weitere Ticker in self.indices unterstützen.
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, relative_error FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung für das Jahr 2020
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'].dt.year == 2020].copy()

            # Umrechnung der relativen Abweichung in Prozent
            df['relative_error_percent'] = df['relative_error'] * 100

            # Gruppierung nach Kalenderwoche (1..53)
            df['week'] = df['date'].dt.isocalendar().week
            grouped = df.groupby('week')['relative_error_percent'].mean()

            # Reindex auf alle 53 möglichen ISO-Wochen (einige Jahre haben 53 Wochen)
            # Falls keine Daten für eine Woche vorliegen, wird 0 gesetzt
            grouped = grouped.reindex(range(1, 54), fill_value=0)
            weekly_data[ticker] = grouped

        # Erstelle die Subplots für SPX (oben) und NDX (unten)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

        # Balkenbreite
        bar_width = 0.6

        # X-Achse: 1..53 für Kalenderwochen
        all_weeks = range(1, 54)

        # Nur jede 4. Woche beschriften (Week 1, 5, 9, ...)
        label_weeks = range(1, 54, 4)

        # --- OBERES DIAGRAMM: S&P 500 (SPX) ---
        for t in self.indices:
            if t.upper() == "SPX":
                spx_2020 = weekly_data.get(t)
                axes[0].bar(
                    all_weeks,
                    spx_2020.values,
                    color='deepskyblue',
                    label='S&P 500',
                    width=bar_width
                )
                axes[0].set_title("Wöchentliche durchschnittliche relative Abweichung für S&P 500 (2020)")
                axes[0].set_ylabel("Relative Abweichung (%)")
                axes[0].legend()
                axes[0].grid(True, linestyle='--', alpha=0.7)

        # --- UNTERES DIAGRAMM: Nasdaq 100 (NDX) ---
        for t in self.indices:
            if t.upper() == "NDX":
                ndx_2020 = weekly_data.get(t)
                axes[1].bar(
                    all_weeks,
                    ndx_2020.values,
                    color='navy',
                    label='Nasdaq 100',
                    width=bar_width
                )
                axes[1].set_title("Wöchentliche durchschnittliche relative Abweichung für Nasdaq 100 (2020)")
                axes[1].set_ylabel("Relative Abweichung (%)")
                axes[1].set_xlabel("Kalenderwoche (1..53)")
                axes[1].legend()
                axes[1].grid(True, linestyle='--', alpha=0.7)

        # X-Achsen-Formatierung: nur alle 4 Wochen beschriften
        for ax in axes:
            ax.set_xticks(label_weeks)
            ax.set_xticklabels([str(w) for w in label_weeks], rotation=0)

        plt.tight_layout()

        # Speichere den Plot im gewünschten Verzeichnis
        save_path = os.path.join(self.plots_dir, "weekly_rel_error_2020.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()

    def monthly_rel_error_for_2020_2022(self):
        """
        Erstellt ein Diagramm mit zwei Subplots (oben: S&P 500, unten: Nasdaq 100),
        das die monatliche durchschnittliche relative Abweichung (in %) für den Zeitraum
        2020 bis 2022 (36 Monate) zeigt.

        Schritte:
          1) 'relative_error' wird in Prozent umgerechnet.
          2) Daten werden für die Jahre 2020 bis 2022 gefiltert.
          3) Gruppierung nach fortlaufendem Monatsindex (0 bis 35) und Mittelwert-Bildung.
          4) Darstellung als Balkendiagramm (x-Achse: 36 Monate, Beschriftung: "Jan 2020", "Feb 2020", …, "Dez 2022").
          5) Für S&P 500 wird die Farbe 'deepskyblue' genutzt, für Nasdaq 100 'navy'.
          6) Es werden keine Zahlen über den Balken angezeigt.

        Der Plot wird als "monthly_rel_error_2020_2022.png" im Plot-Verzeichnis gespeichert.
        """
        # Dictionary zum Speichern der monatlich aggregierten Daten
        monthly_data = {}

        # Fokussiere dich auf SPX (S&P 500) und NDX (Nasdaq 100)
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, relative_error FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung für 2020 bis 2022
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2022)].copy()

            # Umrechnung der relativen Abweichung in Prozent
            df['relative_error_percent'] = df['relative_error'] * 100

            # Fortlaufender Monats-Index: 0 = Jan 2020, 1 = Feb 2020, …, 35 = Dez 2022
            df['yearmonth_index'] = (df['date'].dt.year - 2020) * 12 + (df['date'].dt.month - 1)

            # Gruppierung nach yearmonth_index und Berechnung des Mittelwerts
            grouped = df.groupby('yearmonth_index')['relative_error_percent'].mean()
            grouped = grouped.reindex(range(36), fill_value=0)
            monthly_data[ticker] = grouped

        # Erzeuge eine Liste mit Labels für jeden der 36 Monate
        import calendar
        yearmonth_labels = []
        current_year, current_month = 2020, 1
        for _ in range(36):
            month_name = calendar.month_abbr[current_month]
            yearmonth_labels.append(f"{month_name} {current_year}")
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        # Erstelle die Subplots für SPX (oben) und NDX (unten)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
        x_positions = np.arange(36)
        bar_width = 0.6

        # Plot für S&P 500
        for t in self.indices:
            if t.upper() == "SPX":
                spx_series = monthly_data.get(t)
                axes[0].bar(x_positions, spx_series.values, color='deepskyblue',
                            label='S&P 500', width=bar_width)
                axes[0].set_title("Monatliche durchschnittliche relative Abweichung für S&P 500 (2020-2022)")
                axes[0].set_ylabel("Relative Abweichung (%)")
                axes[0].legend()
                axes[0].grid(True, linestyle='--', alpha=0.7)

        # Plot für Nasdaq 100
        for t in self.indices:
            if t.upper() == "NDX":
                ndx_series = monthly_data.get(t)
                axes[1].bar(x_positions, ndx_series.values, color='navy',
                            label='Nasdaq 100', width=bar_width)
                axes[1].set_title("Monatliche durchschnittliche relative Abweichung für Nasdaq 100 (2020-2022)")
                axes[1].set_ylabel("Relative Abweichung (%)")
                axes[1].set_xlabel("Monate")
                axes[1].legend()
                axes[1].grid(True, linestyle='--', alpha=0.7)

        # X-Achse formatieren
        for ax in axes:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(yearmonth_labels, rotation=45)

        plt.tight_layout()

        # Speichere den Plot im gewünschten Verzeichnis
        save_path = os.path.join(self.plots_dir, "monthly_rel_error_2020_2022.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()
    #endregion

    # ----------------------------------------------------------------
    # 2.14) Plottet die monatliche relative Abweichung nach Moneyness und verbleibenden Tagen (2020)
    # ----------------------------------------------------------------

    #region 2.14) Plottet die monatliche relative Abweichung nach Moneyness und verbleibenden Tagen (2020)
    def monthly_rel_error_moneyness_and_days_2020(self, at_money_tolerance=0.01):
        """
        Erstellt zwei separate Figuren für das Jahr 2020, jeweils mit zwei Subplots:

        Figur A: "Monatliche durchschnittliche relative Abweichung nach Moneyness-Kategorie"
          - Moneyness-Kategorien (3 Linien):
              * im Geld     : moneyness > +at_money_tolerance
              * am Geld     : -at_money_tolerance <= moneyness <= +at_money_tolerance
              * aus dem Geld: moneyness < -at_money_tolerance
          - X-Achse: Monate (Jan–Dez), Y-Achse: relative Abweichung (in %)
          - Zwei Subplots:
              * OBEN: S&P 500
              * UNTEN: Nasdaq 100

        Figur B: "Monatliche durchschnittliche relative Abweichung nach Laufzeit-Kategorie"
          - Laufzeit-Kategorien (3 Linien):
              * 0-30 Tage
              * 31-90 Tage
              * >90 Tage
          - X-Achse: Monate (Jan–Dez), Y-Achse: relative Abweichung (in %)
          - Zwei Subplots:
              * OBEN: S&P 500
              * UNTEN: Nasdaq 100

        Farbgebung:
          - Für Moneyness-Kategorien:
              * im Geld: blau
              * am Geld: grün
              * aus dem Geld: rot
          - Für Laufzeit-Kategorien:
              * 0-30 Tage: blau
              * 31-90 Tage: grün
              * >90 Tage: rot

        Die Diagramme werden aus den Tabellen prepared_{ticker}_data entnommen und
        unter "monthly_rel_error_moneyness_2020.png" bzw. "monthly_rel_error_days_2020.png"
        im Verzeichnis self.plots_dir gespeichert.
        """

        # --- Hilfsfunktionen für Kategorisierung ---
        def categorize_moneyness(m):
            if m > at_money_tolerance:
                return "im Geld"
            elif m < -at_money_tolerance:
                return "aus dem Geld"
            else:
                return "am Geld"

        def categorize_days(d):
            if d <= 30:
                return "0-30 Tage"
            elif d <= 90:
                return "31-90 Tage"
            else:
                return ">90 Tage"

        import calendar

        # Dictionaries für aggregierte Ergebnisse
        data_moneyness = {}
        data_days = {}

        # Daten für Ticker laden (z.B. ["NDX", "SPX"])
        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, relative_error, moneyness, remaining_days FROM {table_name}"
            df = query_db(self.prepared_db_path, query)
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'].dt.year == 2020].copy()
            if df.empty:
                continue
            df['relative_error_percent'] = df['relative_error'] * 100
            df['month'] = df['date'].dt.month

            # --- Moneyness-Kategorisierung ---
            df['moneyness_cat'] = df['moneyness'].apply(categorize_moneyness)
            pivot_m = df.groupby(['month', 'moneyness_cat'])['relative_error_percent'].mean().unstack('moneyness_cat')
            # Gewünschte Reihenfolge: ["im Geld", "am Geld", "aus dem Geld"]
            cat_order_m = ["im Geld", "am Geld", "aus dem Geld"]
            pivot_m = pivot_m.reindex(columns=cat_order_m, fill_value=0)
            pivot_m = pivot_m.reindex(range(1, 13), fill_value=0)
            data_moneyness[ticker.upper()] = pivot_m

            # --- Laufzeit-Kategorisierung ---
            df['days_cat'] = df['remaining_days'].apply(categorize_days)
            pivot_d = df.groupby(['month', 'days_cat'])['relative_error_percent'].mean().unstack('days_cat')
            cat_order_d = ["0-30 Tage", "31-90 Tage", ">90 Tage"]
            pivot_d = pivot_d.reindex(columns=cat_order_d, fill_value=0)
            pivot_d = pivot_d.reindex(range(1, 13), fill_value=0)
            data_days[ticker.upper()] = pivot_d

        # Farbzuordnung festlegen
        # Für Moneyness-Kategorien:
        moneyness_colors = {"im Geld": "blue", "am Geld": "green", "aus dem Geld": "red"}
        # Für Laufzeit-Kategorien:
        days_colors = {"0-30 Tage": "blue", "31-90 Tage": "green", ">90 Tage": "red"}

        # --- FIGUR A: Moneyness-Kategorien ---
        fig_m, axes_m = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
        fig_m.suptitle("Monatliche durchschnittliche relative Abweichung nach Moneyness-Kategorie (2020)", fontsize=12)
        months = np.arange(1, 13)
        month_labels = [calendar.month_abbr[m] for m in months]

        # SPX OBEN, NDX UNTEN (Farbgebung gemäß moneyness_colors)
        if "SPX" in data_moneyness:
            spx_m = data_moneyness["SPX"]
            for cat in spx_m.columns:
                axes_m[0].plot(months, spx_m[cat], marker='o', label=cat, color=moneyness_colors.get(cat))
            axes_m[0].set_title("S&P 500 (2020)")
            axes_m[0].set_ylabel("Relative Abweichung (%)")
            axes_m[0].legend(title="Moneyness-Kategorie")
            axes_m[0].grid(True, linestyle='--', alpha=0.7)
        if "NDX" in data_moneyness:
            ndx_m = data_moneyness["NDX"]
            for cat in ndx_m.columns:
                axes_m[1].plot(months, ndx_m[cat], marker='o', label=cat, color=moneyness_colors.get(cat))
            axes_m[1].set_title("Nasdaq 100 (2020)")
            axes_m[1].set_ylabel("Relative Abweichung (%)")
            axes_m[1].set_xlabel("Monate")
            axes_m[1].legend(title="Moneyness-Kategorie")
            axes_m[1].grid(True, linestyle='--', alpha=0.7)

        for ax in axes_m:
            ax.set_xticks(months)
            ax.set_xticklabels(month_labels, rotation=0)

        fig_m.tight_layout(rect=[0, 0, 1, 0.96])
        save_path_m = os.path.join(self.plots_dir, "monthly_rel_error_moneyness_2020.png")
        fig_m.savefig(save_path_m)
        print(f"Plot gespeichert unter: {save_path_m}")

        # --- FIGUR B: Laufzeit-Kategorien ---
        fig_d, axes_d = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
        fig_d.suptitle("Monatliche durchschnittliche relative Abweichung nach Laufzeit-Kategorie (2020)", fontsize=12)

        # SPX OBEN, NDX UNTEN (Farbgebung gemäß days_colors)
        if "SPX" in data_days:
            spx_d = data_days["SPX"]
            for cat in spx_d.columns:
                axes_d[0].plot(months, spx_d[cat], marker='o', label=cat, color=days_colors.get(cat))
            axes_d[0].set_title("S&P 500 (2020)")
            axes_d[0].set_ylabel("Relative Abweichung (%)")
            axes_d[0].set_xlabel("Monate")
            axes_d[0].legend(title="Laufzeit-Kategorie")
            axes_d[0].grid(True, linestyle='--', alpha=0.7)
        if "NDX" in data_days:
            ndx_d = data_days["NDX"]
            for cat in ndx_d.columns:
                axes_d[1].plot(months, ndx_d[cat], marker='o', label=cat, color=days_colors.get(cat))
            axes_d[1].set_title("Nasdaq 100 (2020)")
            axes_d[1].set_ylabel("Relative Abweichung (%)")
            axes_d[1].legend(title="Laufzeit-Kategorie")
            axes_d[1].grid(True, linestyle='--', alpha=0.7)

        for ax in axes_d:
            ax.set_xticks(months)
            ax.set_xticklabels(month_labels, rotation=0)

        fig_d.tight_layout(rect=[0, 0, 1, 0.96])
        save_path_d = os.path.join(self.plots_dir, "monthly_rel_error_days_2020.png")
        fig_d.savefig(save_path_d)
        print(f"Plot gespeichert unter: {save_path_d}")

        plt.show()
    #endregion

    # ----------------------------------------------------------------
    # 2.15) Analysiert die Moneyness in Abhängigkeit von der Restlauf-Kategorie (Feb.-Apr. 2020)
    # ----------------------------------------------------------------

    #region 2.15) Analysiert die Moneyness in Abhängigkeit von der Restlauf-Kategorie (Feb.-Apr. 2020)
    def analyze_moneyness_by_remaining_days_feb_mar_apr_2020(self, at_money_tolerance=0.01):
        """
        Für die Monate Februar, März und April 2020 berechnet diese Funktion für jeden Ticker in self.indices
        in den drei bekannten Laufzeitgruppen (0-30 Tage, 31-90 Tage, >90 Tage) folgendes:
          1) Den prozentualen Anteil der Optionen, die "aus dem Geld" liegen
             (d.h. Optionen, bei denen der moneyness-Wert < -at_money_tolerance ist).
          2) Den durchschnittlichen moneyness-Wert, jedoch nur für die Optionen, die "aus dem Geld" sind.

        Die Ergebnisse werden für jeden Ticker und jeden Monat (Februar, März, April 2020) formatiert ausgegeben.
        """

        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date, moneyness, remaining_days FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung: Jahr 2020, Monate Februar, März, April
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.year == 2020) & (df['date'].dt.month.isin([2, 3, 4]))].copy()
            if df.empty:
                print(f"Keine Daten für {ticker.upper()} in Feb-Apr 2020.")
                continue

            # Erstelle Spalten: 'month' und 'month_name'
            df['month'] = df['date'].dt.month
            df['month_name'] = df['month'].apply(lambda m: calendar.month_name[m])

            # Laufzeit-Kategorisierung
            def categorize_days(d):
                if d <= 30:
                    return "0-30 Tage"
                elif d <= 90:
                    return "31-90 Tage"
                else:
                    return ">90 Tage"

            df['days_cat'] = df['remaining_days'].apply(categorize_days)

            print(f"\nErgebnisse für {ticker.upper()} (Februar - April 2020):")
            # Für jeden Monat separat
            for month in sorted(df['month'].unique()):
                df_month = df[df['month'] == month]
                month_name = df_month['month_name'].iloc[0]
                print(f"\nMonat: {month_name} 2020")
                # Für jede Laufzeitgruppe:
                for group in ["0-30 Tage", "31-90 Tage", ">90 Tage"]:
                    df_group = df_month[df_month['days_cat'] == group]
                    total_options = len(df_group)
                    if total_options == 0:
                        print(f"  Laufzeitgruppe {group}: Keine Optionen")
                        continue
                    # Filtere "aus dem Geld" Optionen: moneyness < -at_money_tolerance
                    df_aus = df_group[df_group['moneyness'] < -at_money_tolerance]
                    count_aus = len(df_aus)
                    percent_aus = (count_aus / total_options) * 100
                    # Durchschnittliche moneyness nur für "aus dem Geld" Optionen berechnen
                    if count_aus > 0:
                        avg_moneyness = df_aus['moneyness'].mean()
                    else:
                        avg_moneyness = float('nan')
                    print(f"  Laufzeitgruppe {group}: Gesamt Optionen = {total_options}, "
                          f"'aus dem Geld' = {percent_aus:.2f}% , "
                          f"Durchschnittliche Moneyness (aus dem Geld) = {avg_moneyness:.3f}")
            print("-" * 60)
    #endregion

    # ----------------------------------------------------------------
    # 2.16) Plottet die monatliche Gesamtzahl der Contracts (2020)
    # ----------------------------------------------------------------

    #region 2.16) Plottet die monatliche Gesamtzahl der Contracts (2020)
    def monthly_total_contracts_2020(self):
        """
        Erstellt ein Balkendiagramm für das Jahr 2020, das die monatliche Gesamtzahl der Contracts
        für jeden Ticker darstellt. Das Diagramm besteht aus zwei Subplots:
          - Oberer Subplot: S&P 500 (SPX)
          - Unterer Subplot: Nasdaq 100 (NDX)

        Die Daten werden aus den Tabellen prepared_{ticker}_data entnommen, indem die Anzahl der
        Datensätze (Contracts) pro Monat ermittelt wird.

        Der Plot wird als "monthly_total_contracts_2020.png" im Verzeichnis self.plots_dir gespeichert.
        """
        # Dictionary zum Speichern der monatlich aggregierten Contract-Zahlen
        contracts_dict = {}

        for ticker in self.indices:
            table_name = f"prepared_{ticker.lower()}_data"
            query = f"SELECT date FROM {table_name}"
            df = query_db(self.prepared_db_path, query)

            # Datumsformatierung und Filterung für 2020
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'].dt.year == 2020].copy()

            # Spalte 'month' erstellen (1..12)
            df['month'] = df['date'].dt.month

            # Gruppiere nach Monat und zähle die Anzahl der Contracts (Zeilen)
            monthly_counts = df.groupby('month').size()
            # Sicherstellen, dass alle Monate 1..12 vorhanden sind
            monthly_counts = monthly_counts.reindex(range(1, 13), fill_value=0)

            contracts_dict[ticker.upper()] = monthly_counts

            print(f"Monatliche Gesamtzahl der Contracts für {ticker.upper()} (2020):")
            print(monthly_counts)
            print("-" * 50)

        # Erstelle das Balkendiagramm mit zwei Subplots: oben SPX, unten NDX
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
        # Monatsbeschriftungen
        import calendar
        months = np.arange(1, 13)
        month_labels = [calendar.month_abbr[m] for m in months]

        # Oberer Subplot: S&P 500 (SPX)
        if "SPX" in contracts_dict:
            spx_counts = contracts_dict["SPX"]
            axes[0].bar(months, spx_counts.values, color='deepskyblue', width=0.6, label="S&P 500")
            axes[0].set_title("Monatliche Gesamtzahl der Contracts - S&P 500 (2020)")
            axes[0].set_ylabel("Anzahl der Contracts")
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.7)

        # Unterer Subplot: Nasdaq 100 (NDX)
        if "NDX" in contracts_dict:
            ndx_counts = contracts_dict["NDX"]
            axes[1].bar(months, ndx_counts.values, color='navy', width=0.6, label="Nasdaq 100")
            axes[1].set_title("Monatliche Gesamtzahl der Contracts - Nasdaq 100 (2020)")
            axes[1].set_ylabel("Anzahl der Contracts")
            axes[1].set_xlabel("Monate")
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.7)

        # X-Achsen-Beschriftung
        for ax in axes:
            ax.set_xticks(months)
            ax.set_xticklabels(month_labels, rotation=0)

        plt.tight_layout()

        # Speichern
        save_path = os.path.join(self.plots_dir, "monthly_total_contracts_2020.png")
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")

        plt.show()
    #endregion

#endregion
