import sys
from pathlib import Path

# Adiciona a pasta src ao sys.path para importações modulares
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from DataImportExploration import DataImportExploration
from DataCleaningPreProcessing import DataCleaningPreProcessing
from ExploratoryDataAnalysis import ExploratoryDataAnalysis
from FeatureTransformation import FeatureTransformation
from FeatureSelection import FeatureSelection
from FeatureEngineering import FeatureEngineering
from ComparativeAnalysis import ComparativeAnalysis


def main():
    print("Pipeline de Feature Analysis and Engineering - OURO (CLOSE)")

    # ETAPA 1: Data Import and Exploration
    data_explorer = DataImportExploration()
    raw_df = data_explorer.run_exploration(verbose=True)

    # ETAPA 2: Data Cleaning and Preprocessing
    cleaner = DataCleaningPreProcessing(df=raw_df, target_col="Close")
    cleaned_df = cleaner.run_cleaning(verbose=True, save_plots=True)

    # ETAPA 3: Exploratory Data Analysis (EDA)
    eda_analyzer = ExploratoryDataAnalysis(df=cleaned_df, target_col="Close")
    eda_insights = eda_analyzer.run_eda(verbose=True)

    # ETAPA 4: Feature Transformation Techniques
    transformer = FeatureTransformation(df=cleaned_df, target_col="Close")
    transformed_data = transformer.run_transformation(verbose=True, save_plots=True)

    # ETAPA 5: Feature Selection Methods
    selector = FeatureSelection(df=transformed_data["df_hybrid"], target_col="Close")
    selection_results = selector.run_selection(verbose=True, save_plots=True)

    # ETAPA 6: Feature Engineering
    engineer = FeatureEngineering(df_clean=cleaned_df, target_col="Close")
    engineering_results = engineer.run_engineering(verbose=True, save_plots=True)

    # ETAPA 7: Comparative Analysis and Reporting
    comparator = ComparativeAnalysis(
        df_clean=cleaned_df,
        df_engineered=engineering_results["df_engineered"],
        target_col="Close"
    )
    comparative_results = comparator.run_comparison(verbose=True, save_plots=True)

    print("\nPipeline executado com sucesso!")


if __name__ == "__main__":
    main()
