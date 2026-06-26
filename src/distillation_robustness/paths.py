from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TEACHER_OUTPUT_DIR = DATA_DIR / "teacher_outputs"
SCORED_DATA_DIR = DATA_DIR / "scored"
FILTERED_DATA_DIR = DATA_DIR / "filtered"

MODEL_DIR = PROJECT_ROOT / "models"
