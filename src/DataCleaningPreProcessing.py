import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Configurar diretório de cache do Matplotlib para ambiente seguro
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class DataCleaningPreProcessing:
    """
    Etapa 2: Limpeza e Pré-processamento dos Dados (Data Cleaning and Preprocessing)
    
    Esta classe é responsável por:
    - Normalizar e validar a coluna temporal 'Date'.
    - Reconciliar inconsistências lógicas nos limites de preços OHLC (High < Low, Close fora de [Low, High]).
    - Tratar valores ausentes na coluna 'Sales'.
    - Tratar ocorrências de 'Volume == 0'.
    - Justificar tecnicamente a preservação de extremos em variáveis de preço.
    - Gerar métricas e gráficos comparativos de "Antes vs. Depois" da limpeza.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None, target_col: str = "Close"):
        """
        Inicializa o módulo de limpeza e pré-processamento.
        
        :param df: DataFrame pandas bruto (se None, carrega via DataImportExploration).
        :param target_col: Nome da variável-alvo.
        """
        self.target_col = target_col
        if df is None:
            from DataImportExploration import DataImportExploration
            explorer = DataImportExploration()
            self.raw_df = explorer.load_data().copy()
        else:
            self.raw_df = df.copy()

        self.cleaned_df: Optional[pd.DataFrame] = None
        self.cleaning_metrics: Dict[str, Any] = {}

    def parse_and_sort_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte a coluna 'Date' para datetime com interpretação robusta de formatos mistos
        (ex: '30-08-00' vs '1/9/2000'), ordena cronologicamente e reinicia o índice.
        """
        df_out = df.copy()
        if "Date" in df_out.columns:
            df_out["Date"] = pd.to_datetime(df_out["Date"], dayfirst=True, format="mixed")
            df_out = df_out.sort_values(by="Date").reset_index(drop=True)
        return df_out

    def reconcile_ohlc_consistency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Garante a coerência física das barras de negociação financeiras (OHLC):
        - High deve ser >= max(Open, Close, Low)
        - Low deve ser <= min(Open, Close, High)
        
        Corrige os casos onde High < Low e Close fora de [Low, High] sem descartar linhas valiosas de mercado.
        """
        df_out = df.copy()
        price_cols = ["Open", "High", "Low", "Close"]
        
        # Diagnóstico prévio
        initial_high_low_violations = int((df_out["High"] < df_out["Low"]).sum())
        initial_close_out_of_bounds = int(
            ((df_out["Close"] > df_out["High"]) | (df_out["Close"] < df_out["Low"])).sum()
        )

        clean_high = df_out[["High", "Open", "Close"]].max(axis=1)
        clean_low = df_out[["Low", "Open", "Close"]].min(axis=1)

        df_out["High"] = clean_high
        df_out["Low"] = clean_low

        # Diagnóstico pós-ajuste
        post_high_low_violations = int((df_out["High"] < df_out["Low"]).sum())
        post_close_out_of_bounds = int(
            ((df_out["Close"] > df_out["High"]) | (df_out["Close"] < df_out["Low"])).sum()
        )

        metrics = {
            "pre_high_low_violations": initial_high_low_violations,
            "post_high_low_violations": post_high_low_violations,
            "pre_close_bounds_violations": initial_close_out_of_bounds,
            "post_close_bounds_violations": post_close_out_of_bounds,
        }
        return df_out, metrics

    def impute_missing_sales(self, df: pd.DataFrame, method: str = "time_linear") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Trata os 937 valores omissos da coluna 'Sales'.
        Por se tratar de uma série temporal contínua, aplica interpolação linear seguida de
        bfill/ffill para os limites, preservando o tamanho amostral e a suavidade temporal.
        """
        df_out = df.copy()
        pre_nulls = int(df_out["Sales"].isna().sum())

        if method == "time_linear":
            df_out["Sales"] = df_out["Sales"].interpolate(method="linear").bfill().ffill()
        elif method == "median":
            median_val = df_out["Sales"].median()
            df_out["Sales"] = df_out["Sales"].fillna(median_val)
        else:
            raise ValueError(f"Método de imputação não suportado: {method}")

        post_nulls = int(df_out["Sales"].isna().sum())
        metrics = {
            "pre_null_sales": pre_nulls,
            "post_null_sales": post_nulls,
            "imputation_method": method
        }
        return df_out, metrics

    def handle_zero_volume(self, df: pd.DataFrame, method: str = "linear_interp") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Trata as 418 ocorrências de 'Volume == 0'.
        Em mercados financeiros, volume zero reflete fecho de balcão, feriados ou ausência
        temporária de liquidez registada.
        Substitui 0 por NaN e interpola linearmente (com bfill para as primeiras linhas),
        garantindo Volume >= 1 para viabilizar transformações logarítmicas posteriores.
        """
        df_out = df.copy()
        pre_zeros = int((df_out["Volume"] == 0).sum())

        if method == "linear_interp":
            clean_vol = df_out["Volume"].replace(0, np.nan)
            df_out["Volume"] = clean_vol.interpolate(method="linear").bfill()
            # Assegurar valores positivos arredondados
            df_out["Volume"] = df_out["Volume"].round().astype(int)
        elif method == "replace_with_one":
            df_out["Volume"] = df_out["Volume"].replace(0, 1)

        post_zeros = int((df_out["Volume"] == 0).sum())
        metrics = {
            "pre_zero_volume": pre_zeros,
            "post_zero_volume": post_zeros,
            "volume_min_after": int(df_out["Volume"].min())
        }
        return df_out, metrics

    def evaluate_price_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Avalia estatisticamente outliers de preços (Open, High, Low, Close) via Tukey IQR rule.
        Fundamentação Teórica:
        Os valores extremos nas cotações (> 3.261 USD no ouro entre 2024 e 2026) representam
        uma valorização macroeconómica real (tendência secular) e NÃO erro de medição.
        Portanto, o clipping ou remoção de preços destruiria a integridade económica do target,
        sendo preservados intactos para normalização posterior.
        """
        report = {}
        for col in ["Open", "High", "Low", "Close"]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            outlier_count = int((df[col] > upper).sum())
            report[col] = {
                "q1": float(round(q1, 2)),
                "q3": float(round(q3, 2)),
                "upper_iqr_threshold": float(round(upper, 2)),
                "statistical_outliers": outlier_count,
                "max_value": float(round(df[col].max(), 2)),
                "action": "Preservado (tendência secular macroeconómica legítima)"
            }
        return report

    def generate_cleaning_summary(self) -> pd.DataFrame:
        """
        Gera uma tabela estruturada com o resumo comparativo Antes vs Depois da Limpeza.
        """
        if self.cleaned_df is None:
            raise ValueError("O processo de limpeza ainda não foi executado. Chame 'run_cleaning()'.")

        summary_data = [
            {
                "Métrica": "Total de Observações (N)",
                "Antes (Raw)": len(self.raw_df),
                "Depois (Cleaned)": len(self.cleaned_df),
                "Variação": "Sem perda de registos (100% mantido)"
            },
            {
                "Métrica": "Valores Nulos em 'Sales'",
                "Antes (Raw)": int(self.raw_df["Sales"].isna().sum()),
                "Depois (Cleaned)": int(self.cleaned_df["Sales"].isna().sum()),
                "Variação": "-937 (imputação linear temporal)"
            },
            {
                "Métrica": "Ocorrências de Volume == 0",
                "Antes (Raw)": int((self.raw_df["Volume"] == 0).sum()),
                "Depois (Cleaned)": int((self.cleaned_df["Volume"] == 0).sum()),
                "Variação": "-418 (interpolação preservando liquidez)"
            },
            {
                "Métrica": "Violações de High < Low",
                "Antes (Raw)": int((self.raw_df["High"] < self.raw_df["Low"]).sum()),
                "Depois (Cleaned)": int((self.cleaned_df["High"] < self.cleaned_df["Low"]).sum()),
                "Variação": "-1 (reconciliação de limites OHLC)"
            },
            {
                "Métrica": "Close fora de [Low, High]",
                "Antes (Raw)": int(((self.raw_df["Close"] > self.raw_df["High"]) | (self.raw_df["Close"] < self.raw_df["Low"])).sum()),
                "Depois (Cleaned)": int(((self.cleaned_df["Close"] > self.cleaned_df["High"]) | (self.cleaned_df["Close"] < self.cleaned_df["Low"])).sum()),
                "Variação": "-430 (reconciliação de limites OHLC)"
            },
            {
                "Métrica": "Formato de Data",
                "Antes (Raw)": "Object/String mista ('-' e '/')",
                "Depois (Cleaned)": "datetime64[ns] ordenado",
                "Variação": "Cronologia validada"
            }
        ]
        return pd.DataFrame(summary_data)

    def save_cleaning_plots(self, output_path: str = "outputs/cleaning_effects.png"):
        """
        Gera e guarda uma figura multi-painel com os efeitos comparativos Antes vs Depois da limpeza.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        # 1. Gráfico de Valores Omissos (Missing Values)
        labels = ["Raw (Antes)", "Cleaned (Depois)"]
        missing_sales = [self.raw_df["Sales"].isna().sum(), self.cleaned_df["Sales"].isna().sum()]
        colors = ["#d9534f", "#5cb85c"]
        axes[0, 0].bar(labels, missing_sales, color=colors, width=0.5, edgecolor="black")
        axes[0, 0].set_title("(a) Valores Omissos em 'Sales'", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Contagem de NaNs")
        for i, v in enumerate(missing_sales):
            axes[0, 0].text(i, v + 20, str(v), ha="center", fontweight="bold")
        axes[0, 0].set_ylim(0, 1100)

        # 2. Gráfico de Violações de Limites OHLC
        raw_ohlc_issues = (
            (self.raw_df["High"] < self.raw_df["Low"]).sum() +
            ((self.raw_df["Close"] > self.raw_df["High"]) | (self.raw_df["Close"] < self.raw_df["Low"])).sum()
        )
        clean_ohlc_issues = (
            (self.cleaned_df["High"] < self.cleaned_df["Low"]).sum() +
            ((self.cleaned_df["Close"] > self.cleaned_df["High"]) | (self.cleaned_df["Close"] < self.cleaned_df["Low"])).sum()
        )
        axes[0, 1].bar(labels, [raw_ohlc_issues, clean_ohlc_issues], color=colors, width=0.5, edgecolor="black")
        axes[0, 1].set_title("(b) Inconsistências de Limites OHLC", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Total de Violações de Preço")
        axes[0, 1].text(0, raw_ohlc_issues + 10, f"{raw_ohlc_issues} (430 Close + 1 High<Low)", ha="center", fontweight="bold")
        axes[0, 1].text(1, clean_ohlc_issues + 10, str(clean_ohlc_issues), ha="center", fontweight="bold")
        axes[0, 1].set_ylim(0, 500)

        # 3. Gráfico de Contagem de Volume == 0
        zero_vol = [(self.raw_df["Volume"] == 0).sum(), (self.cleaned_df["Volume"] == 0).sum()]
        axes[1, 0].bar(labels, zero_vol, color=colors, width=0.5, edgecolor="black")
        axes[1, 0].set_title("(c) Ocorrências de 'Volume == 0'", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Contagem de Zeros")
        for i, v in enumerate(zero_vol):
            axes[1, 0].text(i, v + 10, str(v), ha="center", fontweight="bold")
        axes[1, 0].set_ylim(0, 500)

        # 4. Amostra de Interpolação de 'Sales' no Tempo
        sample_window = self.raw_df.loc[20:35].copy()
        sample_dates = pd.to_datetime(sample_window["Date"], dayfirst=True, format="mixed")
        clean_window = self.cleaned_df.loc[20:35].copy()
        
        axes[1, 1].plot(sample_dates, clean_window["Sales"], label="Cleaned (Interpolado)", color="#0275d8", linewidth=2, marker="o", markersize=4)
        axes[1, 1].plot(sample_dates, sample_window["Sales"], label="Raw (Original com NaNs)", color="#d9534f", linestyle="--", marker="x", markersize=7)
        axes[1, 1].set_title("(d) Exemplo de Interpolação Temporal em 'Sales'", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("Sales (USD)")
        axes[1, 1].tick_params(axis="x", rotation=30)
        axes[1, 1].legend(loc="best")

        plt.suptitle("Impacto do Pré-processamento e Limpeza de Dados (Antes vs. Depois)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Gráfico de limpeza guardado com sucesso em: {output_file}")

    def run_cleaning(self, verbose: bool = True, save_plots: bool = True) -> pd.DataFrame:
        """
        Executa sequencialmente todas as operações de limpeza e pré-processamento.
        """
        # 1. Parse de Datas
        df_step = self.parse_and_sort_dates(self.raw_df)

        # 2. Reconciliação OHLC
        df_step, ohlc_metrics = self.reconcile_ohlc_consistency(df_step)

        # 3. Imputação de Sales
        df_step, sales_metrics = self.impute_missing_sales(df_step, method="time_linear")

        # 4. Tratamento de Volume == 0
        df_step, volume_metrics = self.handle_zero_volume(df_step, method="linear_interp")

        # 5. Avaliação de Outliers
        outlier_metrics = self.evaluate_price_outliers(df_step)

        self.cleaned_df = df_step

        if save_plots:
            self.save_cleaning_plots("outputs/cleaning_effects.png")

        if verbose:
            print("=" * 80)
            print(" ETAPA 2: DATA CLEANING AND PREPROCESSING")
            print("=" * 80)
            summary_table = self.generate_cleaning_summary()
            print(summary_table.to_string(index=False))
            print("-" * 80)
            print("JUSTIFICAÇÃO DE TRATAMENTO DE OUTLIERS DE PREÇO (Close, Open, High, Low):")
            for col, res in outlier_metrics.items():
                print(f" • {col.ljust(6)}: {res['statistical_outliers']} pontos estatísticos acima de {res['upper_iqr_threshold']} USD (Max: {res['max_value']} USD).")
            print("DECISÃO : Nenhum clipping/remoção de preço aplicado. Os valores extremos refletem a valorização real do ouro.")

        return self.cleaned_df


if __name__ == "__main__":
    cleaner = DataCleaningPreProcessing()
    cleaner.run_cleaning(verbose=True, save_plots=True)
