"""Load the Student Depression dataset."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "student_depression_dataset.csv"


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["?"])
    df.columns = [c.strip() for c in df.columns]
    return df
