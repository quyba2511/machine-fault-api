from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "machine_data.csv"


def generate(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    temperature = np.clip(rng.normal(70, 12, n_samples), 35, 110)
    pressure = np.clip(rng.normal(3.5, 0.8, n_samples), 1.0, 7.0)
    vibration = np.clip(rng.normal(4.0, 1.8, n_samples), 0.2, 12.0)
    run_hours = np.clip(rng.gamma(2.2, 1300, n_samples), 50, 10000)
    alarm_count = np.clip(rng.poisson(1.8, n_samples), 0, 12)
    maintenance_days = np.clip(rng.gamma(2.0, 28, n_samples), 1, 365)

    # A deliberately imperfect fault process: features affect risk, but random
    # noise creates overlap between normal and fault machines.
    score = (
        -4.2
        + 0.055 * (temperature - 70)
        + 0.18 * (pressure - 3.5)
        + 0.38 * (vibration - 4.0)
        + 0.00022 * (run_hours - 2500)
        + 0.32 * alarm_count
        + 0.012 * (maintenance_days - 45)
        + rng.normal(0, 1.15, n_samples)
    )
    fault_probability = 1.0 / (1.0 + np.exp(-score))
    fault = rng.binomial(1, fault_probability)

    return pd.DataFrame(
        {
            "temperature": temperature.round(2),
            "pressure": pressure.round(2),
            "vibration": vibration.round(2),
            "run_hours": run_hours.round(0).astype(int),
            "alarm_count": alarm_count.astype(int),
            "maintenance_days": maintenance_days.round(0).astype(int),
            "fault": fault.astype(int),
        }
    )


def main() -> None:
    df = generate()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(df)} samples at {DATA_PATH}")
    print("Class counts:")
    print(df["fault"].value_counts().sort_index())


if __name__ == "__main__":
    main()
