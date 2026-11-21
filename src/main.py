import sys
from pathlib import Path

# Adiciona a pasta src ao sys.path para importações modulares
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from DataImportExploration import DataImportExploration


def main():
    print("=" * 80)
    print("Pipeline de Feature Analysis and Engineering - OURO (CLOSE)")
    print("=" * 80)

    # ETAPA 1: Data Import and Exploration
    data_explorer = DataImportExploration()
    raw_df = data_explorer.run_exploration(verbose=True)



if __name__ == "__main__":
    main()
