
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

CONFIG_PATH = str(ROOT_DIR / "config")
DATA_DIR = Path(ROOT_DIR / "data")
DATASETS_DIR = Path(ROOT_DIR / "data" / "datasets")
RESULTS_DIR = Path(ROOT_DIR / "results")

FIGURES_DIR = Path(ROOT_DIR / "figures")
if not FIGURES_DIR.exists():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = Path(ROOT_DIR / "checkpoints")

