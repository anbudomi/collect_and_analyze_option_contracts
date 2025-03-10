class DataAnalyzer:
    def __init__(
            self,
            sorted_db_path,
            indices_to_analyze
    ):
        self.sorted_db_path = sorted_db_path
        self.indices_to_analyze = indices_to_analyze

    def load_data(self, index_name):
        """Lädt Optionsdaten für einen bestimmten Index aus der SQLite-Datenbank"""
        table_name = f"sorted_{index_name.lower()}_data"  # Dynamischer Tabellenname
        conn = sqlite3.connect(self.sorted_db_path)

        query = f"""
        SELECT trade_date, execution_price AS strike_price, 
               market_price_base AS spot_price, ticker
        FROM {table_name}
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # Sicherstellen, dass trade_date als Datum erkannt wird
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['month'] = df['trade_date'].dt.to_period('M')  # Monat extrahieren
        return df

    def classify_options(self, df):
        """Berechnet ITM/OTM Status basierend auf Ticker (Call/Put Unterscheidung)"""

        def classify(row):
            ticker = row['ticker'].lower()
            if "c" in ticker:  # Annahme: 'C' für Call, 'P' für Put
                return 'ITM' if row['spot_price'] > row['strike_price'] else 'OTM'
            elif "p" in ticker:
                return 'ITM' if row['spot_price'] < row['strike_price'] else 'OTM'
            else:
                return 'Unknown'

        df['status'] = df.apply(classify, axis=1)
        return df


    def analyze_and_plot(self):
        """Erstellt Diagramme für die ITM/OTM-Verteilung - Gesamt & getrennt nach Call/Put für SPX & NDX"""
        for index in self.indices_to_analyze:
            df = self.load_data(index)
            df = self.classify_options(df)

            # 1) Gesamtübersicht für den Index (wie bisher)
            self.plot_itm_otm_distribution(df, index, f'ITM vs. OTM Verteilung für {index}')

            # 2) Separate Analysen für Call und Put Optionen
            df_call = df[df['ticker'].str.contains(r'C\d{6}', regex=True, case=False)]
            df_put = df[df['ticker'].str.contains(r'P\d{6}', regex=True, case=False)]

            if not df_call.empty:
                self.plot_itm_otm_distribution(df_call, index, f'ITM vs. OTM {index} Call')
            if not df_put.empty:
                self.plot_itm_otm_distribution(df_put, index, f'ITM vs. OTM {index} Put')

    def plot_itm_otm_distribution(self, df, index, title):
        """Erstellt ein gestapeltes Balkendiagramm für die ITM/OTM-Verteilung"""
        monthly_counts = df.groupby(['month', 'status']).size().unstack().fillna(0)
        monthly_percent = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100

        # Plot erstellen
        fig, ax = plt.subplots(figsize=(14, 6))
        monthly_percent.plot(kind='bar', stacked=True, alpha=0.85, ax=ax)

        # Titel & Achsen
        ax.set_title(title)
        ax.set_xlabel('Monat')
        ax.set_ylabel('Prozent')
        ax.legend(title='Status')

        # X-Achse optimieren
        ax.set_xticks(range(0, len(monthly_percent.index), max(1, len(monthly_percent.index) // 12)))
        ax.set_xticklabels(monthly_percent.index[::max(1, len(monthly_percent.index) // 12)], rotation=45, ha="right")

        # Gitterlinien für bessere Lesbarkeit
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Diagramm anzeigen
        plt.show()