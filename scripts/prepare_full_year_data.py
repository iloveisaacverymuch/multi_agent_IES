from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
HOURS_PER_YEAR = 8760


def read_column(filename, column, nrows):
    return pd.read_csv(
        RAW_DATA_DIR / filename,
        usecols=[column],
        nrows=nrows,
        encoding="utf-8",
    )[column].to_numpy(dtype=np.float64)


def normalize(values, area):
    minimum = np.load(PROCESSED_DATA_DIR / f"min_data_{area}.npy")
    maximum = np.load(PROCESSED_DATA_DIR / f"max_data_{area}.npy")
    return (values - minimum) / (maximum - minimum)


def seasonal_test_rows(values):
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    rows = []
    for month in (0, 3, 6, 9):
        start = (sum(month_days[:month]) + 20) * 24
        rows.append(values[start : start + 24])
    return np.vstack(rows)


def build_full_year_arrays():
    electric = read_column("electric_load.csv", "value", 75935)
    cooling = read_column("cooling_commercial.csv", "load", 26280)
    solar = read_column("solar.csv", "value", 26304)

    residential_heat = read_column(
        "residential_heating.csv", "load", HOURS_PER_YEAR
    )
    commercial_heat = read_column(
        "commercial_heating.csv", "load", HOURS_PER_YEAR
    )
    industry_heat = sum(
        read_column(filename, "load", HOURS_PER_YEAR)
        for filename in (
            "chemicals_and_petrochemicals.csv",
            "food_and_tobacco.csv",
            "iron_and_steel.csv",
            "metalic_minerals.csv",
            "paper.csv",
        )
    )

    electricity_price = np.tile(
        [0.4] * 7 + [0.6] * 4 + [1.1] * 3 + [0.6] * 4 + [1.1] * 4 + [0.4] * 2,
        365,
    )

    raw = {
        "residential": np.column_stack(
            (
                electric[67175:75935] / 5.0,
                residential_heat * 1.5,
                cooling[:8760],
                electricity_price,
                solar[:8760],
            )
        ),
        "commercial": np.column_stack(
            (
                electric[:8760] / 5.0,
                commercial_heat * 3.0,
                cooling[8760:17520] * 1.5,
                electricity_price,
                solar[8760:17520],
            )
        ),
        "industry": np.column_stack(
            (
                electric[32855:41615] / 5.0,
                industry_heat * 3.0,
                cooling[17520:26280],
                electricity_price,
                solar[17544:26304],
            )
        ),
    }
    return {area: normalize(values, area) for area, values in raw.items()}


def main():
    arrays = build_full_year_arrays()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for area, values in arrays.items():
        if values.shape != (HOURS_PER_YEAR, 5):
            raise RuntimeError(f"Unexpected {area} full-year shape: {values.shape}")
        existing_test = np.load(PROCESSED_DATA_DIR / f"test_data_{area}.npy")
        representative_days = seasonal_test_rows(values)
        if not np.allclose(representative_days, existing_test, rtol=0.0, atol=1e-12):
            difference = float(np.max(np.abs(representative_days - existing_test)))
            raise RuntimeError(
                f"{area} full-year data do not reproduce the current test set; "
                f"maximum difference={difference}"
            )
        output = PROCESSED_DATA_DIR / f"full_year_data_{area}.npy"
        np.save(output, values)
        print(f"saved {output.relative_to(PROJECT_ROOT)} shape={values.shape}")


if __name__ == "__main__":
    main()
