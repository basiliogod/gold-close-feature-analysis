import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class DataImportExploration:
    """
    Etapa 1: Importação e Exploração dos Dados (Data Import and Exploration)
    
    Esta classe é responsável por:
    - Carregar o dataset bruto (Gold_data.csv).
    - Descrever a estrutura dimensional (observações, variáveis, uso de memória).
    - Formalizar a separação entre Variável-Alvo ('Close') e Variáveis de Entrada (Features).
    - Gerar tabelas sumárias estatísticas (head, describe, info).
    - Analisar a distribuição estatística inicial da variável alvo ('Close').
    - Detetar irregularidades preliminares (valores omissos, zeros anómalos, tipos e formatos).
    """

    def __init__(self, file_path: Optional[str] = None, target_col: str = "Close"):
        """
        Inicializa o explorador de dados.

        :param file_path: Caminho para o ficheiro CSV. Se None, resolve para data/Gold_data.csv.
        :param target_col: Nome da variável-alvo.
        """
        if file_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.file_path = base_dir / "data" / "Gold_data.csv"
        else:
            self.file_path = Path(file_path)

        self.target_col = target_col
        self.df: Optional[pd.DataFrame] = None
        self.features: list = []

    def load_data(self) -> pd.DataFrame:
        """
        Carrega o ficheiro CSV para um DataFrame pandas e valida a sua existência.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Ficheiro não encontrado: {self.file_path}")

        self.df = pd.read_csv(self.file_path)
        
        # Identificar features de entrada
        self.features = [col for col in self.df.columns if col != self.target_col]
        return self.df

    def get_dataset_overview(self) -> Dict[str, Any]:
        """
        Retorna um dicionário com a visão geral do dataset:
        dimensões, target, features, tipos e contagem de nulos/duplicados.
        """
        if self.df is None:
            raise ValueError("Os dados ainda não foram carregados. Chame 'load_data()' primeiro.")

        num_observations, num_variables = self.df.shape
        missing_per_col = self.df.isnull().sum()
        total_missing = missing_per_col.sum()
        duplicated_rows = self.df.duplicated().sum()

        overview = {
            "num_observations": num_observations,
            "num_variables": num_variables,
            "target_variable": self.target_col,
            "input_features": self.features,
            "total_missing_values": int(total_missing),
            "missing_by_column": missing_per_col.to_dict(),
            "duplicated_rows": int(duplicated_rows),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
        return overview

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Gera a tabela de estatísticas descritivas (count, mean, std, min, percentis, max).
        """
        if self.df is None:
            raise ValueError("Os dados ainda não foram carregados. Chame 'load_data()' primeiro.")

        return self.df.describe().T

    def get_target_distribution_stats(self) -> Dict[str, float]:
        """
        Calcula as principais estatísticas descritivas para a variável-alvo (Close).
        """
        if self.df is None:
            raise ValueError("Os dados ainda não foram carregados. Chame 'load_data()' primeiro.")

        if self.target_col not in self.df.columns:
            raise KeyError(f"Variável alvo '{self.target_col}' não existe no dataset.")

        target_series = self.df[self.target_col].dropna()
        
        stats = {
            "count": float(target_series.count()),
            "mean": float(target_series.mean()),
            "std": float(target_series.std()),
            "min": float(target_series.min()),
            "25%": float(target_series.quantile(0.25)),
            "median (50%)": float(target_series.median()),
            "75%": float(target_series.quantile(0.75)),
            "max": float(target_series.max()),
            "skewness": float(target_series.skew()),
            "kurtosis": float(target_series.kurtosis())
        }
        return stats

    def detect_initial_irregularities(self) -> Dict[str, Any]:
        """
        Deteta anomalias e irregularidades iniciais nos dados brutos:
        - Colunas com valores nulos e respetivas percentagens.
        - Ocorrências de zeros em colunas onde zero é suspeito.
        - Incoerências lógicas entre preços (ex: High < Low, Close < Low, Close > High).
        - Heterogeneidade no formato das datas.
        """
        if self.df is None:
            raise ValueError("Os dados ainda não foram carregados. Chame 'load_data()' primeiro.")

        irregularities = {}

        # 1. Valores Omissos
        null_counts = self.df.isnull().sum()
        cols_with_nulls = {
            col: {"count": int(count), "percentage": round((count / len(self.df)) * 100, 2)}
            for col, count in null_counts.items() if count > 0
        }
        irregularities["missing_values"] = cols_with_nulls

        # 2. Valores Zero (particularmente anómalo em Volume ou Preços)
        numeric_cols = self.df.select_dtypes(include=["number"]).columns
        zeros_count = {
            col: int((self.df[col] == 0).sum())
            for col in numeric_cols if (self.df[col] == 0).sum() > 0
        }
        irregularities["zero_values"] = zeros_count

        # 3. Incoerências Lógicas de Preço
        price_cols = {"Open", "High", "Low", "Close"}.intersection(set(self.df.columns))
        if {"High", "Low"}.issubset(price_cols):
            invalid_high_low = int((self.df["High"] < self.df["Low"]).sum())
            irregularities["invalid_high_low_count"] = invalid_high_low

        if {"Close", "High", "Low"}.issubset(price_cols):
            invalid_close_range = int(((self.df["Close"] > self.df["High"]) | (self.df["Close"] < self.df["Low"])).sum())
            irregularities["invalid_close_out_of_bounds_count"] = invalid_close_range

        # 4. Formato das datas
        if "Date" in self.df.columns:
            date_samples = self.df["Date"].dropna().astype(str)
            # Verificar se há separadores mistos ('-' vs '/')
            contains_dash = date_samples.str.contains("-").sum()
            contains_slash = date_samples.str.contains("/").sum()
            irregularities["date_format_mixture"] = {
                "hyphen_separated_dates": int(contains_dash),
                "slash_separated_dates": int(contains_slash)
            }

        return irregularities

    def run_exploration(self, verbose: bool = True) -> pd.DataFrame:
        """
        Executa o fluxo completo da Etapa 1 e imprime o relatório estruturado.
        
        :param verbose: Se True, imprime no terminal os outputs formatados.
        :return: O DataFrame carregado.
        """
        df = self.load_data()
        overview = self.get_dataset_overview()
        desc = self.get_summary_statistics()
        target_stats = self.get_target_distribution_stats()
        irreg = self.detect_initial_irregularities()

        if verbose:
            print("=" * 80)
            print(" ETAPA 1: DATA IMPORT AND EXPLORATION")
            print("=" * 80)
            print(f"Ficheiro carregado: {self.file_path.name}")
            print(f"Dimensões: {overview['num_observations']} observações x {overview['num_variables']} variáveis")
            print(f"Linhas duplicadas: {overview['duplicated_rows']}")
            print("-" * 80)
            
            print(f"\n[1] DEFINIÇÃO DE VARIÁVEIS:")
            print(f" • Variável Alvo (Target): '{self.target_col}'")
            print(f" • Variáveis de Entrada (Features): {self.features}")
            
            print(f"\n[2] AMOSTRA DOS DADOS BRUTOS:")
            print(df.head(5).to_string())
            
            print(f"\n[3] ESTATÍSTICAS DESCRITIVAS GERAIS:")
            print(desc.round(2).to_string())

            print(f"\n[4] DISTRIBUIÇÃO ESTATÍSTICA DO TARGET ('{self.target_col}'):")
            for k, v in target_stats.items():
                print(f" • {k.ljust(15)}: {v:.4f}")

            print(f"\n[5] IRREGULARIDADES IDENTIFICADAS:")
            if irreg.get("missing_values"):
                print(" • Valores Ausentes:")
                for col, data in irreg["missing_values"].items():
                    print(f"    - '{col}': {data['count']} nulos ({data['percentage']}% do dataset)")
            else:
                print(" • Sem valores ausentes detetados.")

            if irreg.get("zero_values"):
                print(" • Ocorrências de Zero em Variáveis Numéricas:")
                for col, count in irreg["zero_values"].items():
                    print(f"    - '{col}': {count} ocorrências com valor 0")

            if irreg.get("invalid_high_low_count", 0) > 0 or irreg.get("invalid_close_out_of_bounds_count", 0) > 0:
                print(f" • Incoerências de Preços: High < Low: {irreg.get('invalid_high_low_count')}, Close fora de [Low, High]: {irreg.get('invalid_close_out_of_bounds_count')}")
            else:
                print(" • Coerência de Preços (High >= Close/Open >= Low): Verificada sem anomalias lógicas.")

            if irreg.get("date_format_mixture"):
                mixture = irreg["date_format_mixture"]
                print(f" • Heterogeneidade na coluna 'Date': {mixture['hyphen_separated_dates']} datas com '-' e {mixture['slash_separated_dates']} datas com '/'.")

        return df


if __name__ == "__main__":
    explorer = DataImportExploration()
    explorer.run_exploration(verbose=True)
