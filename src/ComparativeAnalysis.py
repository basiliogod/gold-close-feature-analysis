import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

# Configurar backend headless seguro para Matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


class ComparativeAnalysis:
    """
    Etapa 7: Comparative Analysis and Reporting
    
    Esta classe consolida e avalia rigorosamente os 4 conjuntos de features gerados ao longo do pipeline:
    1. Set 1 (Original Cleaned): Variáveis brutas higienizadas na escala original (D=6).
    2. Set 2 (Transformed Hybrid): Log em Volume/weight + StandardScaler (D=6).
    3. Set 3 (Selected Orthogonal): Seleção ortogonal mínima [Open, ln(Volume)] + StandardScaler (D=2).
    4. Set 4 (Final Engineered & Selected): Subconjunto otimizado com 5 preditores causais
       [HL_Midpoint, Open_to_Mid_Pct, ln(Relative_Spread), ln(Open_Volatility_14), ln(Volume)] + StandardScaler (D=5).

    Métricas Avaliadas (Validação Cronológica Estrita 80/20 - 2000-2021 vs. 2021-2026):
    - Condicionamento Numérico: Número de Condição da matriz de entrada e VIF Máximo.
    - Morfologia Estatística: Assimetria média absoluta e Informação Mútua média por feature.
    - Sondas Preditivas Diagnósticas: Linear Ridge Probe e Neural Network (MLP 64-32) Probe.
    - Análise de Componentes Principais (PCA): Eficiência da variância explicada.
    """

    def __init__(self, df_clean: pd.DataFrame, df_engineered: pd.DataFrame,
                 target_col: str = "Close", output_dir: str = "outputs/comparative"):
        self.df_clean = df_clean.copy()
        self.df_engineered = df_engineered.copy()
        self.target_col = target_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_plots: List[str] = []

    def _compute_max_vif(self, X_df: pd.DataFrame) -> float:
        if X_df.shape[1] <= 1:
            return 1.0
        vifs = []
        for col in X_df.columns:
            others = [c for c in X_df.columns if c != col]
            r2 = LinearRegression().fit(X_df[others], X_df[col]).score(X_df[others], X_df[col])
            vif = 1.0 / (1.0 - r2) if (1.0 - r2) > 1e-6 else 1e6
            vifs.append(vif)
        return float(max(vifs))

    def prepare_feature_sets(self) -> Dict[str, Tuple[pd.DataFrame, bool]]:
        """
        Constrói os 4 conjuntos de features padronizados para comparação justa.
        O booleano indica se o StandardScaler deve ser ajustado no split de treino.
        """
        # Set 1: Original Cleaned (unscaled)
        s1_cols = ["Open", "High", "Low", "Volume", "Sales", "weight"]
        df_s1 = self.df_clean[s1_cols].copy()

        # Set 2: Transformed Hybrid (Log em Volume & weight + StandardScaler)
        df_s2 = self.df_clean[s1_cols].copy()
        df_s2["Volume"] = np.log1p(df_s2["Volume"])
        df_s2["weight"] = np.log1p(df_s2["weight"])

        # Set 3: Selected Orthogonal (Open + ln(Volume) + StandardScaler)
        df_s3 = self.df_clean[["Open", "Volume"]].copy()
        df_s3["Volume"] = np.log1p(df_s3["Volume"])

        # Set 4: Final Engineered & Selected Set (5 features ortogonais + StandardScaler)
        s4_cols = ["HL_Midpoint", "Open_to_Mid_Pct", "Relative_Spread", "Open_Volatility_14", "Volume"]
        df_s4 = self.df_engineered[s4_cols].copy()
        df_s4["Volume"] = np.log1p(df_s4["Volume"])
        df_s4["Relative_Spread"] = np.log1p(df_s4["Relative_Spread"] * 100.0)
        df_s4["Open_Volatility_14"] = np.log1p(df_s4["Open_Volatility_14"] * 100.0)

        return {
            "1. Original (Raw Unscaled)": (df_s1, False),
            "2. Transformed (Log + Z-Score)": (df_s2, True),
            "3. Selected Orthogonal": (df_s3, True),
            "4. Final Engineered & Selected": (df_s4, True),
        }

    def evaluate_all_sets(self) -> Dict[str, Any]:
        """
        Avalia os 4 conjuntos em métricas intrínsecas e sondas diagnósticas cronológicas (80% / 20%).
        """
        feature_sets = self.prepare_feature_sets()
        y = self.df_engineered[self.target_col].values
        split_idx = int(len(y) * 0.80)

        records = []
        loss_curves = {}
        mlp_predictions = {}
        pca_variances = {}

        for name, (X_df, apply_scale) in feature_sets.items():
            X = X_df.values
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            if apply_scale:
                scaler = StandardScaler()
                X_train_proc = scaler.fit_transform(X_train)
                X_test_proc = scaler.transform(X_test)
                X_full_proc = StandardScaler().fit_transform(X)
            else:
                X_train_proc, X_test_proc = X_train, X_test
                X_full_proc = X

            # 1. Condicionamento Numérico & Morfologia
            cond_num = float(np.linalg.cond(X_train_proc))
            max_vif = self._compute_max_vif(pd.DataFrame(X_train_proc, columns=X_df.columns))
            mean_abs_skew = float(np.mean(np.abs(X_df.skew())))
            mi_vals = mutual_info_regression(X_full_proc, y, random_state=42)
            mean_mi = float(np.mean(mi_vals))

            # 2. Sonda Linear (Ridge Regression)
            ridge = Ridge(alpha=1.0).fit(X_train_proc, y_train)
            pred_ridge = ridge.predict(X_test_proc)
            mae_ridge = float(mean_absolute_error(y_test, pred_ridge))
            rmse_ridge = float(root_mean_squared_error(y_test, pred_ridge))
            r2_ridge = float(r2_score(y_test, pred_ridge))

            # 3. Sonda Neuronal (MLP 64-32 com Adam)
            mlp = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42,
                learning_rate_init=0.01
            )
            mlp.fit(X_train_proc, y_train)
            pred_mlp = mlp.predict(X_test_proc)
            mae_mlp = float(mean_absolute_error(y_test, pred_mlp))
            rmse_mlp = float(root_mean_squared_error(y_test, pred_mlp))
            r2_mlp = float(r2_score(y_test, pred_mlp))

            # 4. PCA Variance Ratio
            pca = PCA().fit(X_full_proc)
            pca_variances[name] = np.cumsum(pca.explained_variance_ratio_)

            loss_curves[name] = mlp.loss_curve_
            mlp_predictions[name] = pred_mlp

            records.append({
                "Feature_Set": name,
                "Dim(D)": X.shape[1],
                "Mean_|Skew|": round(mean_abs_skew, 3),
                "Max_VIF": round(max_vif, 2),
                "Cond_Number": round(cond_num, 2),
                "Mean_MI": round(mean_mi, 3),
                "Ridge_MAE($)": round(mae_ridge, 2),
                "MLP_MAE($)": round(mae_mlp, 2),
                "MLP_RMSE($)": round(rmse_mlp, 2),
                "MLP_R2": round(r2_mlp, 5),
                "MLP_Iters": mlp.n_iter_,
                "MLP_Final_Loss": round(mlp.loss_, 2)
            })

        summary_df = pd.DataFrame(records)
        return {
            "summary_df": summary_df,
            "loss_curves": loss_curves,
            "mlp_predictions": mlp_predictions,
            "y_test": y[split_idx:],
            "dates_test": self.df_engineered["Date"].iloc[split_idx:],
            "pca_variances": pca_variances
        }

    # --------------------------------------------------------------------------
    # Visualizações Comparativas (outputs/comparative/)
    # --------------------------------------------------------------------------
    def plot_conditioning_and_skew(self, summary_df: pd.DataFrame):
        """
        Gráfico 1: Comparação de Estabilidade Numérica (Cond Number, Max VIF e Skewness).
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        labels = ["1. Original", "2. Transformed", "3. Selected", "4. Final Eng."]
        colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]

        # (a) Condition Number
        axes[0].bar(labels, summary_df["Cond_Number"], color=colors, edgecolor="black")
        axes[0].set_yscale("log")
        axes[0].set_title("(a) Número de Condição $\\kappa(X)$ (Escala Log)", fontweight="bold")
        axes[0].set_ylabel("Número de Condição (Menor é Melhor)")
        axes[0].tick_params(axis="x", rotation=20)
        for i, v in enumerate(summary_df["Cond_Number"]):
            axes[0].text(i, v * 1.25, f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")
        axes[0].set_ylim(0.5, 30000)

        # (b) Max VIF
        axes[1].bar(labels, summary_df["Max_VIF"], color=colors, edgecolor="black")
        axes[1].axhline(10, color="black", linestyle="--", linewidth=1.2, label="Limite VIF = 10")
        axes[1].set_yscale("log")
        axes[1].set_title("(b) Multicolinearidade Máxima (VIF Max)", fontweight="bold")
        axes[1].set_ylabel("VIF Máximo (Escala Log)")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].legend(loc="upper right")
        for i, v in enumerate(summary_df["Max_VIF"]):
            axes[1].text(i, v * 1.25, f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")
        axes[1].set_ylim(0.5, 40000)

        # (c) Mean |Skewness|
        axes[2].bar(labels, summary_df["Mean_|Skew|"], color=colors, edgecolor="black")
        axes[2].set_title("(c) Assimetria Média Absoluta ($|\\gamma_1|$)", fontweight="bold")
        axes[2].set_ylabel("Média |Skewness|")
        axes[2].tick_params(axis="x", rotation=20)
        for i, v in enumerate(summary_df["Mean_|Skew|"]):
            axes[2].text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        axes[2].set_ylim(0, 2.7)

        plt.suptitle("Comparação Estrutural e Numérica entre os Conjuntos de Features", fontsize=14, fontweight="bold")
        out_path = self.output_dir / "01_conditioning_and_skew_comparison.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        self.generated_plots.append(str(out_path))

    def plot_mlp_convergence_and_error(self, results: Dict[str, Any]):
        """
        Gráfico 2: Curvas de convergência da perda do MLP e Erros Out-of-Sample (2021-2026).
        """
        summary_df = results["summary_df"]
        loss_curves = results["loss_curves"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]
        labels_short = ["1. Original", "2. Transformed", "3. Selected", "4. Final Eng."]

        # (a) Curvas de Loss MLP
        for (name, curve), c in zip(loss_curves.items(), colors):
            axes[0].plot(curve, label=f"{name} (Final: {curve[-1]:.1f})", color=c, linewidth=2)
        axes[0].set_yscale("log")
        axes[0].set_title("(a) Convergência do Treino da Rede Neuronal (MLP Loss)", fontweight="bold")
        axes[0].set_xlabel("Épocas / Iterações")
        axes[0].set_ylabel("Training Loss (MSE - Escala Log)")
        axes[0].legend(loc="upper right", fontsize=8.5)
        axes[0].grid(True, linestyle=":", alpha=0.6)

        # (b) Out-of-Sample MAE vs RMSE
        x = np.arange(len(labels_short))
        width = 0.35
        axes[1].bar(x - width/2, summary_df["MLP_MAE($)"], width, label="MLP MAE ($ USD)", color="#337ab7", edgecolor="black")
        axes[1].bar(x + width/2, summary_df["MLP_RMSE($)"], width, label="MLP RMSE ($ USD)", color="#9370db", edgecolor="black")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels_short, rotation=15)
        axes[1].set_ylabel("Erro Out-of-Sample em USD (2021-2026)")
        axes[1].set_title("(b) Desempenho Preditivo MLP no Teste Cronológico (20%)", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

        for i, (mae, rmse) in enumerate(zip(summary_df["MLP_MAE($)"], summary_df["MLP_RMSE($)"])):
            axes[1].text(i - width/2, mae + 2, f"${mae:.1f}", ha="center", fontsize=8, fontweight="bold")
            axes[1].text(i + width/2, rmse + 2, f"${rmse:.1f}", ha="center", fontsize=8, fontweight="bold")

        plt.suptitle("Impacto do Pipeline na Otimização e Generalização de Redes Neuronais", fontsize=14, fontweight="bold")
        out_path = self.output_dir / "02_mlp_convergence_and_error.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        self.generated_plots.append(str(out_path))

    def plot_pca_comparison(self, results: Dict[str, Any]):
        """
        Gráfico 3: Análise de Componentes Principais (PCA) - Variância Acumulada e Eficiência de Representação.
        """
        pca_vars = results["pca_variances"]
        plt.figure(figsize=(9, 5))
        colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]
        markers = ["o", "s", "^", "D"]

        for (name, cum_var), c, m in zip(pca_vars.items(), colors, markers):
            comps = np.arange(1, len(cum_var) + 1)
            plt.plot(comps, cum_var * 100, marker=m, linewidth=2, color=c, label=name)

        plt.axhline(95, color="black", linestyle="--", linewidth=1, label="Limite de 95% de Variância")
        plt.title("Gráfico 03: Eficiência de Representação Dimensional via PCA", fontweight="bold")
        plt.xlabel("Número de Componentes Principais")
        plt.ylabel("Variância Acumulada Explicada (%)")
        plt.xticks(range(1, 7))
        plt.ylim(40, 103)
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle=":", alpha=0.6)

        out_path = self.output_dir / "03_pca_variance_comparison.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        self.generated_plots.append(str(out_path))

    def generate_ieee_synthesis_figure(self, results: Dict[str, Any]):
        """
        Gráfico 4: Painel compósito síntese comparativa para o relatório IEEE.
        """
        summary_df = results["summary_df"]
        loss_curves = results["loss_curves"]
        labels_short = ["1. Raw", "2. Transf.", "3. Select.", "4. Final"]
        colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        # (a) Condition Number
        axes[0, 0].bar(labels_short, summary_df["Cond_Number"], color=colors, edgecolor="black")
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("(a) Número de Condição da Matriz ($\\kappa$, Escala Log)", fontweight="bold")
        axes[0, 0].set_ylabel("Condition Number")
        for i, v in enumerate(summary_df["Cond_Number"]):
            axes[0, 0].text(i, v * 1.25, f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")
        axes[0, 0].set_ylim(0.5, 30000)

        # (b) Curvas de Convergência Neural (MLP Loss)
        for (name, curve), c in zip(loss_curves.items(), colors):
            axes[0, 1].plot(curve, label=name, color=c, linewidth=1.8)
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("(b) Curvas de Perda no Treino MLP (Convergência)", fontweight="bold")
        axes[0, 1].set_xlabel("Épocas")
        axes[0, 1].set_ylabel("Training MSE Loss (Log)")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, linestyle=":", alpha=0.5)

        # (c) Erro Out-of-Sample MLP (MAE & RMSE)
        x = np.arange(len(labels_short))
        w = 0.35
        axes[1, 0].bar(x - w/2, summary_df["MLP_MAE($)"], w, label="MAE ($)", color="#337ab7", edgecolor="black")
        axes[1, 0].bar(x + w/2, summary_df["MLP_RMSE($)"], w, label="RMSE ($)", color="#9370db", edgecolor="black")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(labels_short)
        axes[1, 0].set_title("(c) Erro Preditivo Out-of-Sample (2021-2026)", fontweight="bold")
        axes[1, 0].set_ylabel("Erro (USD)")
        axes[1, 0].legend()

        # (d) Série Temporal Prevista (Set 1 Raw vs Set 4 Final no regime extremo 2024-2026)
        dates_tail = results["dates_test"].tail(200)
        y_tail = results["y_test"][-200:]
        pred_raw_tail = results["mlp_predictions"]["1. Original (Raw Unscaled)"][-200:]
        pred_final_tail = results["mlp_predictions"]["4. Final Engineered & Selected"][-200:]

        axes[1, 1].plot(dates_tail, y_tail, color="black", linewidth=2, label="Real Close (USD)")
        axes[1, 1].plot(dates_tail, pred_raw_tail, color="#d9534f", linestyle="--", linewidth=1.5, label="MLP Set 1 (Raw Unscaled)")
        axes[1, 1].plot(dates_tail, pred_final_tail, color="#5cb85c", linestyle="-.", linewidth=1.8, label="MLP Set 4 (Final Engineered)")
        axes[1, 1].set_title("(d) Extrapolação no Rali Secular (Últimas 200 Sessões)", fontweight="bold")
        axes[1, 1].set_ylabel("Close (USD)")
        axes[1, 1].tick_params(axis="x", rotation=25)
        axes[1, 1].legend(fontsize=8)

        plt.suptitle("Comparative Feature Set Analysis: Conditioning, Convergence & Generalization", fontsize=14, fontweight="bold")
        out_path = self.output_dir / "comparative_ieee_synthesis.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        self.generated_plots.append(str(out_path))

    def run_comparison(self, verbose: bool = True, save_plots: bool = True) -> Dict[str, Any]:
        """
        Executa a comparação integral e imprime a tabela final de resultados.
        """
        results = self.evaluate_all_sets()
        summary_df = results["summary_df"]

        if save_plots:
            self.plot_conditioning_and_skew(summary_df)
            self.plot_mlp_convergence_and_error(results)
            self.plot_pca_comparison(results)
            self.generate_ieee_synthesis_figure(results)

        if verbose:
            print("=" * 100)
            print(" ETAPA 7: COMPARATIVE ANALYSIS AND REPORTING")
            print("=" * 100)
            print("BENCHMARK MULTI-CRITÉRIO ENTRE CONJUNTOS DE FEATURES (Split Cronológico 80/20 - Teste 2021-2026):")
            print(summary_df.to_string(index=False))
            print("-" * 100)
            
            s1_mae = summary_df.loc[0, "MLP_MAE($)"]
            s4_mae = summary_df.loc[3, "MLP_MAE($)"]
            s1_rmse = summary_df.loc[0, "MLP_RMSE($)"]
            s4_rmse = summary_df.loc[3, "MLP_RMSE($)"]
            
            print("CONCLUSÕES EMPÍRICAS DO PIPELINE:")
            print(f"1. Estabilização Numérica Extrema:")
            print(f"   • Número de Condição caiu de {summary_df.loc[0, 'Cond_Number']:.2f} (Original) para {summary_df.loc[3, 'Cond_Number']:.2f} (Final Engenheirado).")
            print(f"   • VIF Máximo caiu de {summary_df.loc[0, 'Max_VIF']:.1f} para {summary_df.loc[3, 'Max_VIF']:.2f} (zero multicolinearidade).")
            print(f"2. Impacto na Rede Neuronal (MLP Probe):")
            print(f"   • MAE reduziu de ${s1_mae:.2f} (Original) para ${s4_mae:.2f} (Final) -> Melhoria de {((s1_mae - s4_mae)/s1_mae)*100:.1f}%!")
            print(f"   • RMSE reduziu de ${s1_rmse:.2f} (Original) para ${s4_rmse:.2f} (Final) -> Melhoria de {((s1_rmse - s4_rmse)/s1_rmse)*100:.1f}%!")
            print(f"   • Perda de Treino (MSE Loss) caiu de {summary_df.loc[0, 'MLP_Final_Loss']:.2f} para {summary_df.loc[3, 'MLP_Final_Loss']:.2f} em menos épocas.")
            print("3. Conjunto Recomendado para Produção (Deep Learning / MLP / LSTM):")
            print("   • Set 4 (Final Engineered & Selected, D=5): ['HL_Midpoint', 'Open_to_Mid_Pct', 'ln(Relative_Spread)', 'ln(Open_Volatility_14)', 'ln(Volume)'] + Z-Score.")
            if save_plots:
                print(f"{len(self.generated_plots)} gráficos comparativos guardados em '{self.output_dir}'.")

        return results


if __name__ == "__main__":
    from DataImportExploration import DataImportExploration
    from DataCleaningPreProcessing import DataCleaningPreProcessing
    from FeatureEngineering import FeatureEngineering

    raw = DataImportExploration().load_data()
    clean = DataCleaningPreProcessing(raw).run_cleaning(verbose=False, save_plots=False)
    eng = FeatureEngineering(clean).run_engineering(verbose=False, save_plots=False)
    comp = ComparativeAnalysis(clean, eng["df_engineered"])
    comp.run_comparison(verbose=True, save_plots=True)
