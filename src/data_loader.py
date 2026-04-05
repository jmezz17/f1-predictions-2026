import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import logging
logging.getLogger("fastf1").setLevel(logging.WARNING)

# Point FastF1 to our local cache folder
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def get_race_results(year: int, round_number: int) -> pd.DataFrame:
    """
    Load race results for a given year and round.
    Returns a cleaned DataFrame with one row per driver.
    """
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=False, weather=False, messages=False)

        results = session.results[[
            "DriverNumber", "Abbreviation", "FullName",
            "TeamName", "GridPosition", "Position", "Status", "Points"
        ]].copy()

        results.rename(columns={"Abbreviation": "Driver"}, inplace=True)

        # Convert finish position to numeric (DNFs become NaN)
        results["Position"] = pd.to_numeric(results["Position"], errors="coerce")
        results["GridPosition"] = pd.to_numeric(results["GridPosition"], errors="coerce")

        # Extract race time in seconds for finishers
        results["RaceTime"] = np.nan
        for idx, row in results.iterrows():
            try:
                driver_laps = session.laps.pick_drivers(row["DriverNumber"])
                if not driver_laps.empty:
                    total_time = driver_laps["LapTime"].sum()
                    results.at[idx, "RaceTime"] = total_time.total_seconds()
            except Exception:
                pass

        results["Round"] = round_number
        results["Year"] = year
        results["EventName"] = session.event["EventName"]

        return results.reset_index(drop=True)

    except Exception as e:
        print(f"  Could not load race {year} round {round_number}: {e}")
        return pd.DataFrame()


def get_quali_results(year: int, round_number: int) -> pd.DataFrame:
    """
    Load qualifying results for a given year and round.
    Returns a DataFrame with one row per driver, including best quali time.
    """
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load(telemetry=False, weather=False, messages=False)

        results = session.results[[
            "DriverNumber", "Abbreviation", "FullName",
            "TeamName", "Q1", "Q2", "Q3", "Position"
        ]].copy()

        results.rename(columns={
            "Abbreviation": "Driver",
            "Position": "GridPosition"
        }, inplace=True)

        # Best quali time: use Q3, fall back to Q2, then Q1
        def best_time(row):
            for col in ["Q3", "Q2", "Q1"]:
                val = row[col]
                if pd.notna(val) and hasattr(val, "total_seconds"):
                    return val.total_seconds()
            return np.nan

        results["QualiTime"] = results.apply(best_time, axis=1)
        results["GridPosition"] = pd.to_numeric(results["GridPosition"], errors="coerce")
        results["Round"] = round_number
        results["Year"] = year

        return results[["DriverNumber", "Driver", "FullName", "TeamName",
                         "QualiTime", "GridPosition", "Round", "Year"]].reset_index(drop=True)

    except Exception as e:
        print(f"  Could not load quali {year} round {round_number}: {e}")
        return pd.DataFrame()


def get_full_season(year: int, max_rounds: int = 24) -> dict:
    """
    Load all completed race results for a season.
    Returns a dict of {round_number: DataFrame}.
    Skips rounds that haven't happened yet or fail to load.
    """
    season = {}
    print(f"Loading {year} season...")

    for round_num in range(1, max_rounds + 1):
        print(f"  Round {round_num}...", end=" ")
        df = get_race_results(year, round_num)
        if not df.empty:
            season[round_num] = df
            print(f"✓ {df['EventName'].iloc[0]}")
        else:
            print("skipped")

    print(f"Loaded {len(season)} rounds for {year}\n")
    return season


if __name__ == "__main__":
    # Quick test - load 2024 round 1
    print("Testing data loader...")
    print("\n--- 2024 Round 1 Race Results ---")
    race = get_race_results(2024, 1)
    if not race.empty:
        print(race[["Driver", "FullName", "TeamName",
                     "GridPosition", "Position", "RaceTime"]].to_string(index=False))
        print(f"\nMAE test - loaded {len(race)} drivers")

    print("\n--- 2024 Round 1 Qualifying ---")
    quali = get_quali_results(2024, 1)
    if not quali.empty:
        print(quali[["Driver", "TeamName", "QualiTime",
                      "GridPosition"]].to_string(index=False))