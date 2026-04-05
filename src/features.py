import pandas as pd
import numpy as np

# Team tier mapping - reflects constructor competitiveness
# 1 = top, 2 = midfield, 3 = backmarker
TEAM_TIERS = {
    # Tier 1 - race winners
    "Mercedes": 1,
    "Ferrari": 1,
    "McLaren": 1,
    "Red Bull Racing": 1,

    # Tier 2 - competitive midfield
    "Haas F1 Team": 2,
    "Haas": 2,
    "TGR Haas F1 Team": 2,  # 2026 official name
    "Alpine": 2,
    "Racing Bulls": 2,
    "Williams": 2,
    "Aston Martin": 2,

    # Tier 3 - backmarkers
    "Audi": 3,
    "Cadillac": 3,

    # Historical name variations for training data
    "AlphaTauri": 2,
    "RB": 2,
    "Visa Cash App RB": 2,
    "Kick Sauber": 3,
    "Sauber": 3,
    "Alfa Romeo": 2,
}

# 2026 F1 calendar circuit types (with Bahrain and KSA canceled)
CIRCUIT_TYPES = {
    1:  "technical",  # Australia - Albert Park
    2:  "technical",  # China - Shanghai
    3:  "technical",  # Japan - Suzuka
    4:  "street",     # Miami
    5:  "street",     # Canada - Montreal semi-street
    6:  "street",     # Monaco
    7:  "technical",  # Barcelona-Catalunya
    8:  "power",      # Austria - Red Bull Ring
    9:  "technical",  # Great Britain - Silverstone
    10: "power",      # Belgium - Spa
    11: "technical",  # Hungary - Hungaroring
    12: "technical",  # Netherlands - Zandvoort
    13: "power",      # Italy - Monza
    14: "street",     # Spain - Madrid street circuit (new!)
    15: "street",     # Azerbaijan - Baku
    16: "street",     # Singapore - Marina Bay
    17: "technical",  # USA - Austin
    18: "street",     # Mexico City
    19: "technical",  # Brazil - Interlagos
    20: "street",     # Las Vegas
    21: "street",     # Qatar - Lusail
    22: "street",     # Abu Dhabi - Yas Marina
}

# Rookies in 2026
ROOKIES_2026 = [
    "LIN",  # Arvid Lindblad - Racing Bulls
]


def get_team_tier(team_name: str) -> int:
    """Return team tier (1=top, 2=mid, 3=back). Default 2 if unknown."""
    return TEAM_TIERS.get(team_name, 2)


def get_circuit_type(round_number: int) -> str:
    """Return circuit type for a given round."""
    return CIRCUIT_TYPES.get(round_number, "technical")


def encode_circuit_type(circuit_type: str) -> int:
    """Encode circuit type as integer for the model."""
    mapping = {"street": 0, "power": 1, "technical": 2}
    return mapping.get(circuit_type, 2)


def compute_driver_form(races_list: list, driver: str, last_n: int = 3) -> float:
    """
    Compute average finishing position over last N races.
    DNFs count as P20. Returns NaN if no data available.
    """
    positions = []
    for race_df in races_list[-last_n:]:
        driver_row = race_df[race_df["Driver"] == driver]
        if not driver_row.empty:
            pos = driver_row["Position"].values[0]
            positions.append(pos if not np.isnan(pos) else 20.0)

    return np.mean(positions) if positions else np.nan


def build_feature_matrix(
    races_dict: dict,
    quali_df: pd.DataFrame,
    round_number: int,
    year: int = 2026
) -> pd.DataFrame:
    """
    Build the feature matrix for a given race round.
    Each row is a driver, columns are features + targets.

    Features:
        - grid_position: qualifying position
        - team_tier: 1/2/3
        - circuit_type_encoded: 0/1/2
        - driver_form: avg finish position last 3 races
        - is_rookie: 0/1
        - quali_time: best qualifying lap in seconds

    Targets (for training):
        - target_position: actual finishing position
        - target_race_time: actual race time in seconds
    """
    circuit_type = get_circuit_type(round_number)
    circuit_encoded = encode_circuit_type(circuit_type)

    # Get ordered list of past races for form calculation
    past_rounds = sorted([r for r in races_dict.keys() if r < round_number])
    past_races = [races_dict[r] for r in past_rounds]

    rows = []
    for _, driver_row in quali_df.iterrows():
        driver = driver_row["Driver"]
        team = driver_row["TeamName"]

        feature = {
            "Driver": driver,
            "FullName": driver_row.get("FullName", driver),
            "TeamName": team,
            "grid_position": driver_row["GridPosition"],
            "quali_time": driver_row.get("QualiTime", np.nan),
            "team_tier": get_team_tier(team),
            "circuit_type_encoded": circuit_encoded,
            "driver_form": compute_driver_form(past_races, driver),
            "is_rookie": 1 if driver in ROOKIES_2026 else 0,
        }

        # Add targets if this round exists in races_dict (training mode)
        if round_number in races_dict:
            race_df = races_dict[round_number]
            race_row = race_df[race_df["Driver"] == driver]
            if not race_row.empty:
                feature["target_position"] = race_row["Position"].values[0]
                feature["target_race_time"] = race_row["RaceTime"].values[0]
            else:
                feature["target_position"] = np.nan
                feature["target_race_time"] = np.nan
        else:
            # Prediction mode - no targets yet
            feature["target_position"] = np.nan
            feature["target_race_time"] = np.nan

        rows.append(feature)

    return pd.DataFrame(rows).reset_index(drop=True)


if __name__ == "__main__":
    from src.data_loader import get_race_results, get_quali_results
    import logging
    logging.getLogger("fastf1").setLevel(logging.WARNING)

    print("Testing feature engineering...")

    # Load a few 2024 races
    races = {}
    for r in [1, 2, 3]:
        df = get_race_results(2024, r)
        if not df.empty:
            races[r] = df

    # Load quali for round 3
    quali = get_quali_results(2024, 3)

    # Build feature matrix for round 3
    features = build_feature_matrix(races, quali, round_number=3, year=2024)

    print("\nFeature matrix for 2024 Round 3:")
    print(features[[
        "Driver", "TeamName", "grid_position", "team_tier",
        "circuit_type_encoded", "driver_form", "is_rookie",
        "target_position"
    ]].to_string(index=False))