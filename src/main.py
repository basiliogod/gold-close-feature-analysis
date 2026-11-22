import sys
from pathlib import Path

# Adiciona a pasta src ao sys.path para importações modulares
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from DataImportExploration import DataImportExploration
from DataCleaningPreProcessing import DataCleaningPreProcessing


def main():
    print("Pipeline de Feature Analysis and Engineering - OURO (CLOSE)")

    # ETAPA 1: Data Import and Exploration
    data_explorer = DataImportExploration()
    raw_df = data_explorer.run_exploration(verbose=True)

    # ETAPA 2: Data Cleaning and Preprocessing
    cleaner = DataCleaningPreProcessing(df=raw_df, target_col="Close")
    cleaned_df = cleaner.run_cleaning(verbose=True, save_plots=True)

if __name__ == "__main__":
    main()
