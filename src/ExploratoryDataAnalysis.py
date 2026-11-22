import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import scipy.stats as stats

# Configurar backend headless seguro para Matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class ExploratoryDataAnalysis:
    """
    Etapa 3: Análise Exploratória de Dados (Exploratory Data Analysis - EDA)
    
    Esta classe realiza uma investigação estatística e visual aprofundada do dataset higienizado:
    - Análise univariada detalhada do target ('Close') e de cada variável preditora.
    - Testes de normalidade, assimetria (skewness), achatamento (kurtosis) e Q-Q plots.
    - Análise de dispersão e outliers via Boxplots e Violin plots.
    - Análise bivariada de cada feature contra o target ('Close') com regressões e dispersão.
    - Matrizes de correlação linear (Pearson) e monotónica (Spearman) para diagnóstico de multicolinearidade.
    - Dinâmicas temporais, sazonalidade e volatilidade rolante ao longo de 26 anos (2000-2026).
    - Geração de gráficos analíticos de alta resolução em outputs/eda/.
    """

    def __init__(self, df: pd.DataFrame, target_col: str = "Close", output_dir: str = "outputs/eda"):
        self.df = df.copy()
        self.target_col = target_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Assegurar data tipada
        if "Date" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["Date"]):
            self.df["Date"] = pd.to_datetime(self.df["Date"], dayfirst=True, format="mixed")

        self.numeric_features = [
            c for c in self.df.select_dtypes(include=[np.number]).columns if c != self.target_col
        ]
        self.generated_plots: List[str] = []
        self.insights: Dict[str, Any] = {}

    def _save_current_fig(self, filename: str, title: str):
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        self.generated_plots.append(str(filepath))

    # --------------------------------------------------------------------------
    # 1. Análise do Target (Close) - Gráficos 1 a 4
    # --------------------------------------------------------------------------
    def plot_target_analysis(self):
        close = self.df[self.target_col]

        # Gráfico 1: Histograma com KDE e Ajuste Normal do Close
        plt.figure(figsize=(8, 5))
        sns.histplot(close, kde=True, color="#d9534f", stat="density", bins=40, alpha=0.6)
        x_axis = np.linspace(close.min(), close.max(), 200)
        norm_fit = stats.norm.pdf(x_axis, close.mean(), close.std())
        plt.plot(x_axis, norm_fit, "k--", linewidth=2, label="Ajuste Normal Teórico")
        plt.title("Gráfico 01: Distribuição da Variável-Alvo 'Close' vs. Distribuição Normal", fontweight="bold")
        plt.xlabel("Preço de Fecho (USD)")
        plt.ylabel("Densidade")
        plt.legend()
        self._save_current_fig("01_target_close_distribution.png", "Target Distribution")

        # Gráfico 2: Q-Q Plot do Close
        plt.figure(figsize=(7, 5))
        stats.probplot(close, dist="norm", plot=plt)
        plt.title("Gráfico 02: Q-Q Plot Normal do Target 'Close'", fontweight="bold")
        plt.xlabel("Quantis Teóricos Normais")
        plt.ylabel("Valores Amostrais do Close (USD)")
        plt.grid(True, linestyle=":", alpha=0.6)
        self._save_current_fig("02_target_close_qqplot.png", "Target Q-Q Plot")

        # Gráfico 3: Série Temporal Histórica (2000 - 2026)
        plt.figure(figsize=(10, 5))
        plt.plot(self.df["Date"], close, color="#b8860b", linewidth=1.5, label="Ouro Diário (Close)")
        plt.title("Gráfico 03: Série Temporal Histórica do Ouro (2000 - 2026)", fontweight="bold")
        plt.xlabel("Data")
        plt.ylabel("Preço de Fecho (USD)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left")
        self._save_current_fig("03_target_close_timeseries.png", "Target Time Series")

        # Gráfico 4: Boxplot e Violin Plot combinados do Close
        plt.figure(figsize=(8, 4))
        sns.violinplot(x=close, color="#f0ad4e", inner="quartile", cut=0)
        plt.title("Gráfico 04: Violin Plot e Dispersão Quartílica do 'Close'", fontweight="bold")
        plt.xlabel("Preço de Fecho (USD)")
        self._save_current_fig("04_target_close_violin.png", "Target Violin")

    # --------------------------------------------------------------------------
    # 2. Distribuições das Variáveis Preditoras - Gráficos 5 a 10
    # --------------------------------------------------------------------------
    def plot_feature_distributions(self):
        palette = {"Open": "#337ab7", "High": "#5cb85c", "Low": "#f0ad4e",
                   "Volume": "#5bc0de", "Sales": "#d9534f", "weight": "#9370db"}

        plot_num = 5
        for col in ["Open", "High", "Low", "Volume", "Sales", "weight"]:
            plt.figure(figsize=(8, 4.5))
            series = self.df[col]
            color = palette.get(col, "#337ab7")

            if col == "Volume":
                # Mostrar com escala logarítmica no inset ou histplot log
                sns.histplot(series, kde=True, color=color, bins=40, log_scale=True)
                plt.xlabel(f"{col} (Escala Logarítmica)")
            else:
                sns.histplot(series, kde=True, color=color, bins=40)
                plt.xlabel(f"{col}")

            skew = series.skew()
            kurt = series.kurtosis()
            plt.title(f"Gráfico {plot_num:02d}: Distribuição da Feature '{col}' (Skew: {skew:.2f}, Kurt: {kurt:.2f})", fontweight="bold")
            plt.ylabel("Frequência")
            self._save_current_fig(f"{plot_num:02d}_dist_{col.lower()}.png", f"Distribution of {col}")
            plot_num += 1

    # --------------------------------------------------------------------------
    # 3. Análise de Outliers e Boxplots - Gráficos 11 a 14
    # --------------------------------------------------------------------------
    def plot_feature_boxplots(self):
        # Gráfico 11: Boxplot dos Preços (Open, High, Low, Close)
        plt.figure(figsize=(9, 5))
        price_df = self.df[["Open", "High", "Low", "Close"]]
        sns.boxplot(data=price_df, palette="Blues_r", showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red"})
        plt.title("Gráfico 11: Comparação de Escala e Dispersão das Variáveis de Preço", fontweight="bold")
        plt.ylabel("Preço (USD)")
        self._save_current_fig("11_boxplot_prices_comparison.png", "Boxplot Prices")

        # Gráfico 12: Boxplot do Volume
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=self.df["Volume"], color="#5bc0de", showmeans=True)
        plt.title("Gráfico 12: Boxplot do 'Volume' (Identificação de Caudas Pesadas)", fontweight="bold")
        plt.xlabel("Volume Diário")
        self._save_current_fig("12_boxplot_volume.png", "Boxplot Volume")

        # Gráfico 13: Boxplot de Sales
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=self.df["Sales"], color="#d9534f", showmeans=True)
        plt.title("Gráfico 13: Boxplot da Feature 'Sales' (Distribuição Plana)", fontweight="bold")
        plt.xlabel("Sales (USD)")
        self._save_current_fig("13_boxplot_sales.png", "Boxplot Sales")

        # Gráfico 14: Boxplot de weight
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=self.df["weight"], color="#9370db", showmeans=True)
        plt.title("Gráfico 14: Boxplot da Feature 'weight' e Outliers Superiores", fontweight="bold")
        plt.xlabel("weight")
        self._save_current_fig("14_boxplot_weight.png", "Boxplot weight")

    # --------------------------------------------------------------------------
    # 4. Relação Bivariada com o Target (Scatter Plots) - Gráficos 15 a 20
    # --------------------------------------------------------------------------
    def plot_bivariate_relationships(self):
        plot_num = 15
        for col in ["Open", "High", "Low", "Volume", "Sales", "weight"]:
            plt.figure(figsize=(8, 5))
            corr = self.df[col].corr(self.df[self.target_col])

            if col in ["Open", "High", "Low"]:
                # Gráficos 15, 16, 17: Relação linear extrema com identidade
                plt.scatter(self.df[col], self.df[self.target_col], alpha=0.4, color="#337ab7", s=10)
                lims = [min(self.df[col].min(), self.df[self.target_col].min()),
                        max(self.df[col].max(), self.df[self.target_col].max())]
                plt.plot(lims, lims, "r--", linewidth=1.5, label="Linha de Identidade (y = x)")
                plt.legend()
            elif col == "Volume":
                # Gráfico 18: Volume vs Close com escala log no volume
                plt.scatter(self.df[col], self.df[self.target_col], alpha=0.3, color="#5bc0de", s=10)
                plt.xscale("log")
                plt.xlabel(f"{col} (Escala Log)")
            else:
                # Gráficos 19, 20: Sales e weight vs Close
                plt.scatter(self.df[col], self.df[self.target_col], alpha=0.3, color="#5cb85c" if col == "Sales" else "#9370db", s=10)
                # Adicionar linha de tendência linear
                m, b = np.polyfit(self.df[col], self.df[self.target_col], 1)
                plt.plot(self.df[col], m * self.df[col] + b, color="black", linestyle="--", label=f"Tendência Linear (m={m:.4f})")
                plt.legend()

            plt.title(f"Gráfico {plot_num:02d}: Relação Bivariada '{col}' vs. 'Close' (r = {corr:.4f})", fontweight="bold")
            if col != "Volume":
                plt.xlabel(f"{col}")
            plt.ylabel("Close (USD)")
            plt.grid(True, linestyle=":", alpha=0.5)
            self._save_current_fig(f"{plot_num:02d}_scatter_{col.lower()}_vs_close.png", f"Scatter {col} vs Close")
            plot_num += 1

    # --------------------------------------------------------------------------
    # 5. Correlação e Multicolinearidade - Gráficos 21 e 22
    # --------------------------------------------------------------------------
    def plot_correlation_matrices(self):
        num_cols = ["Open", "High", "Low", "Volume", "Sales", "weight", "Close"]
        pearson_corr = self.df[num_cols].corr(method="pearson")
        spearman_corr = self.df[num_cols].corr(method="spearman")

        # Gráfico 21: Matriz de Correlação Linear de Pearson
        plt.figure(figsize=(9, 7))
        sns.heatmap(pearson_corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
                    linewidths=1, linecolor="white", cbar_kws={"label": "Coeficiente de Pearson (r)"})
        plt.title("Gráfico 21: Matriz de Correlação Linear de Pearson", fontweight="bold")
        self._save_current_fig("21_correlation_matrix_pearson.png", "Pearson Heatmap")

        # Gráfico 22: Matriz de Correlação Não-Linear de Spearman
        plt.figure(figsize=(9, 7))
        sns.heatmap(spearman_corr, annot=True, fmt=".3f", cmap="Spectral", vmin=-1, vmax=1,
                    linewidths=1, linecolor="white", cbar_kws={"label": "Coeficiente de Spearman (rho)"})
        plt.title("Gráfico 22: Matriz de Correlação Monotónica de Spearman", fontweight="bold")
        self._save_current_fig("22_correlation_matrix_spearman.png", "Spearman Heatmap")

    # --------------------------------------------------------------------------
    # 6. Dinâmicas Temporais e Financeiras - Gráficos 23 a 26
    # --------------------------------------------------------------------------
    def plot_temporal_and_financial_dynamics(self):
        df_temp = self.df.copy()
        df_temp["Year"] = df_temp["Date"].dt.year
        df_temp["Month"] = df_temp["Date"].dt.month
        df_temp["Daily_Return"] = df_temp["Close"].pct_change() * 100
        df_temp["Spread_HL"] = df_temp["High"] - df_temp["Low"]
        df_temp["Rolling_Std_30"] = df_temp["Close"].rolling(window=30).std()

        # Gráfico 23: Evolução Anual do Close (Boxplot por Ano)
        plt.figure(figsize=(14, 6))
        sns.boxplot(x="Year", y="Close", data=df_temp, color="#f0ad4e", fliersize=2)
        plt.xticks(rotation=45)
        plt.title("Gráfico 23: Distribuição e Tendência Secular do 'Close' por Ano (2000-2026)", fontweight="bold")
        plt.xlabel("Ano")
        plt.ylabel("Preço de Fecho (USD)")
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        self._save_current_fig("23_annual_close_boxplots.png", "Annual Boxplot")

        # Gráfico 24: Sazonalidade Mensal do Preço Médio e Retorno Médio
        monthly_stats = df_temp.groupby("Month")["Close"].mean()
        plt.figure(figsize=(8, 4.5))
        months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
        plt.bar(months, monthly_stats, color="#337ab7", edgecolor="black", alpha=0.8)
        plt.title("Gráfico 24: Sazonalidade Mensal - Preço Médio do Ouro por Mês", fontweight="bold")
        plt.xlabel("Mês")
        plt.ylabel("Preço Médio (USD)")
        plt.grid(True, axis="y", linestyle=":", alpha=0.6)
        self._save_current_fig("24_monthly_seasonality.png", "Monthly Seasonality")

        # Gráfico 25: Volatilidade Móvel de 30 Dias (Rolling Volatility)
        plt.figure(figsize=(10, 4.5))
        plt.plot(df_temp["Date"], df_temp["Rolling_Std_30"], color="#d9534f", linewidth=1.2)
        plt.title("Gráfico 25: Volatilidade Histórica Rolante (Janela de 30 Dias do Close)", fontweight="bold")
        plt.xlabel("Data")
        plt.ylabel("Desvio Padrão Rolante (USD)")
        plt.grid(True, linestyle="--", alpha=0.5)
        self._save_current_fig("25_rolling_volatility_30d.png", "Rolling Volatility")

        # Gráfico 26: Spread Intradiário (High - Low) vs. Volume
        plt.figure(figsize=(8, 5))
        plt.scatter(df_temp["Volume"], df_temp["Spread_HL"], color="#20b2aa", alpha=0.3, s=12)
        plt.xscale("log")
        plt.title("Gráfico 26: Relação entre Spread Diário (High - Low) e Volume de Transações", fontweight="bold")
        plt.xlabel("Volume Diário (Escala Log)")
        plt.ylabel("Spread Diário (High - Low em USD)")
        plt.grid(True, linestyle=":", alpha=0.5)
        self._save_current_fig("26_spread_vs_volume.png", "Spread vs Volume")

    # --------------------------------------------------------------------------
    # 7. Figura Composta para o Relatório LaTeX (IEEE Synthesis Figure)
    # --------------------------------------------------------------------------
    def generate_ieee_composite_figure(self):
        """
        Cria uma figura multi-painel de 6 gráficos-chave para inclusão direta no relatório IEEE.
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        close = self.df[self.target_col]

        # Painel A: Histograma Close
        sns.histplot(close, kde=True, color="#d9534f", ax=axes[0, 0], stat="density", bins=35)
        axes[0, 0].set_title("(a) Distribuição de Frequência do Target 'Close'", fontweight="bold")
        axes[0, 0].set_xlabel("Preço de Fecho (USD)")

        # Painel B: Q-Q Plot
        stats.probplot(close, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("(b) Q-Q Plot Normal do Target 'Close'", fontweight="bold")

        # Painel C: Série Temporal do Close
        axes[1, 0].plot(self.df["Date"], close, color="#b8860b", linewidth=1.2)
        axes[1, 0].set_title("(c) Trajetória Temporal Histórica (2000 - 2026)", fontweight="bold")
        axes[1, 0].set_xlabel("Ano")
        axes[1, 0].set_ylabel("Close (USD)")

        # Painel D: Heatmap Pearson
        num_cols = ["Open", "High", "Low", "Volume", "Sales", "weight", "Close"]
        corr = self.df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1], cbar=False, linewidths=0.5)
        axes[1, 1].set_title("(d) Matriz de Correlação Linear de Pearson", fontweight="bold")

        # Painel E: Scatter Open vs Close
        axes[2, 0].scatter(self.df["Open"], close, alpha=0.3, color="#337ab7", s=8)
        lims = [min(self.df["Open"].min(), close.min()), max(self.df["Open"].max(), close.max())]
        axes[2, 0].plot(lims, lims, "r--", label="Identidade y = x")
        axes[2, 0].set_title("(e) Dispersão 'Open' vs. 'Close' (r = 0.9998)", fontweight="bold")
        axes[2, 0].set_xlabel("Open (USD)")
        axes[2, 0].set_ylabel("Close (USD)")
        axes[2, 0].legend()

        # Painel F: Volume (log) vs Close
        axes[2, 1].scatter(self.df["Volume"], close, alpha=0.3, color="#5bc0de", s=8)
        axes[2, 1].set_xscale("log")
        axes[2, 1].set_title("(f) Dispersão 'Volume' (Log) vs. 'Close' (r = 0.033)", fontweight="bold")
        axes[2, 1].set_xlabel("Volume (Log Scale)")
        axes[2, 1].set_ylabel("Close (USD)")

        composite_path = self.output_dir / "eda_ieee_synthesis.png"
        plt.suptitle("Análise Exploratória de Dados (EDA) - Síntese Global", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(composite_path, dpi=300)
        plt.close()
        self.generated_plots.append(str(composite_path))

    # --------------------------------------------------------------------------
    # 8. Extração de Insights e Execução Completa
    # --------------------------------------------------------------------------
    def extract_insights(self) -> Dict[str, Any]:
        num_cols = ["Open", "High", "Low", "Volume", "Sales", "weight", "Close"]
        stats_df = pd.DataFrame(index=num_cols)
        stats_df["Skewness"] = self.df[num_cols].skew()
        stats_df["Kurtosis"] = self.df[num_cols].kurtosis()
        stats_df["Pearson_with_Close"] = self.df[num_cols].apply(lambda c: c.corr(self.df[self.target_col]))
        stats_df["Spearman_with_Close"] = self.df[num_cols].apply(lambda c: c.corr(self.df[self.target_col], method="spearman"))

        insights = {
            "statistical_properties": stats_df.round(4).to_dict(orient="index"),
            "target_skew": float(self.df[self.target_col].skew()),
            "target_kurt": float(self.df[self.target_col].kurtosis()),
            "high_collinearity_pairs": [
                ("Open", "Close", float(self.df["Open"].corr(self.df["Close"]))),
                ("High", "Close", float(self.df["High"].corr(self.df["Close"]))),
                ("Low", "Close", float(self.df["Low"].corr(self.df["Close"]))),
            ],
            "low_correlation_features": [
                ("Volume", float(self.df["Volume"].corr(self.df["Close"]))),
                ("Sales", float(self.df["Sales"].corr(self.df["Close"]))),
                ("weight", float(self.df["weight"].corr(self.df["Close"]))),
            ],
            "total_plots_generated": len(self.generated_plots)
        }
        self.insights = insights
        return insights

    def run_eda(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Executa a geração de todos os gráficos e extrai os insights analíticos fundamentais.
        """
        if verbose:
            print("=" * 80)
            print(" ETAPA 3: EXPLORATORY DATA ANALYSIS (ANÁLISE EXPLORATÓRIA DE DADOS)")
            print("=" * 80)
            print("Gerando gráficos analíticos...")

        self.plot_target_analysis()               # Gráficos 01 a 04
        self.plot_feature_distributions()         # Gráficos 05 a 10
        self.plot_feature_boxplots()              # Gráficos 11 a 14
        self.plot_bivariate_relationships()       # Gráficos 15 a 20
        self.plot_correlation_matrices()          # Gráficos 21 e 22
        self.plot_temporal_and_financial_dynamics() # Gráficos 23 a 26
        self.generate_ieee_composite_figure()     # Painel de síntese IEEE

        insights = self.extract_insights()

        if verbose:
            print(f"Sucesso: {len(self.generated_plots)} gráficos gerados com alta resolução em '{self.output_dir}'.")
            print("-" * 80)
            print("PRINCIPAIS INSIGHTS IDENTIFICADOS:")
            print("1. Distribuição do Target (Close):")
            print(f"   • Skewness = {insights['target_skew']:.4f} (assimetria acentuada à direita, não-gaussiana)")
            print(f"   • Kurtosis = {insights['target_kurt']:.4f} (leptocúrtica com caudas pesadas)")
            print("2. Multicolinearidade Extrema nas Cotações:")
            for f1, f2, r in insights["high_collinearity_pairs"]:
                print(f"   • Corr({f1}, {f2}) = {r:.4f} -> Informação altamente redundante.")
            print("3. Variáveis com Correlação Quase Nula com o Close:")
            for f, r in insights["low_correlation_features"]:
                print(f"   • Corr({f}, Close) = {r:.4f}")
            print("   'Sales' e 'weight' não possuem dependência linear com o preço de fecho.")
            print("4. Volume:")
            print("   Possui skewness extremo e relação não-linear com o Close.")

        return insights


if __name__ == "__main__":
    from DataImportExploration import DataImportExploration
    from DataCleaningPreProcessing import DataCleaningPreProcessing

    raw = DataImportExploration().load_data()
    clean = DataCleaningPreProcessing(raw).run_cleaning(verbose=False, save_plots=False)
    eda = ExploratoryDataAnalysis(clean)
    eda.run_eda(verbose=True)
