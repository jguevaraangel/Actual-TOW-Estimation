from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from matplotlib.colors import Normalize
from catboost import Pool
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from atow_estimation.config import (
    ARTIFACT_SUBFOLDER_FIGURES,
    FIGURE_FEATURE_IMPORTANCE,
    FIGURE_PARETO,
    FIGURE_SCATTER,
    FIGURE_SCATTER_RESIDUALS,
    FIGURE_RESIDUALS_DIST,
    CSV_LEADERBOARD,
    ARTIFACT_SUBFOLDER_LEADERBOARD,
    MODEL_LOGGERS,
)
from atow_estimation.paths import FIGURES_DIR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pyodbc
import getpass
import os
import mlflow
import h2o

# ignore specific SQL warning
warnings.filterwarnings(
    "ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable"
)
# earth radius in nautical miles
EARTH_RADIUS_NM = 3440.065


# utils
def haversine(lat1, lon1, lat2, lon2):
    """Compute great-circle distance using the Haversine formula."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_NM * atan2(sqrt(a), sqrt(1 - a))


def vectorized_haversine(lat, lon, lat_next, lon_next):
    """Vectorized implementation of the Haversine formula."""
    lat, lon, lat_next, lon_next = map(np.radians, [lat, lon, lat_next, lon_next])
    valid = ~np.isnan(lat) & ~np.isnan(lon) & ~np.isnan(lat_next) & ~np.isnan(lon_next)

    distances = np.full(lat.shape, np.nan, dtype=float)
    dlat, dlon = lat_next[valid] - lat[valid], lon_next[valid] - lon[valid]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[valid]) * np.cos(lat_next[valid]) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances[valid] = EARTH_RADIUS_NM * c
    return distances


def bound_trace(takeoff_time, landing_time, df):
    """Bound the trace by ATOT and ALDT."""
    return df[(df["timestamp"] >= takeoff_time) & (df["timestamp"] <= landing_time)].sort_values(
        "timestamp"
    )


def get_flight_series(df, id):
    """Convert df row to series."""
    return df[df["flightKey"] == id].squeeze(axis=0)


def calculate_total_distance(base_df, flight_id, df, threshold=15):
    """Compute the total flight distance based on trace points."""
    flight = get_flight_series(base_df, flight_id)
    df["timestamp"] = pd.to_datetime(
        df["dateReference"].astype(str) + " " + df["time"].astype(str), format="%Y-%m-%d %H:%M:%S"
    )
    df = bound_trace(flight["ATOT_track"], flight["ALDT_track"], df)

    lat, lon = df["lat"].values, df["lng"].values
    lat_next, lon_next = np.append(lat[1:], np.nan), np.append(lon[1:], np.nan)
    df["segment_distance"] = vectorized_haversine(lat, lon, lat_next, lon_next)

    total_distance = df["segment_distance"].sum()

    # gap at the beginning:
    first_point_time = df.iloc[0]["timestamp"]
    gap_start = abs((first_point_time - flight["ATOT_track"]).total_seconds() / 60)
    if gap_start > threshold:
        # distance from departure airport to the first track point.
        extra_start = haversine(
            flight["ADEPLat"], flight["ADEPLong"], df.iloc[0]["lat"], df.iloc[0]["lng"]
        )
        total_distance += extra_start

    # gap at the end:
    last_point_time = df.iloc[-1]["timestamp"]
    gap_end = abs((flight["ALDT_track"] - last_point_time).total_seconds() / 60)
    if gap_end > threshold:
        # distance from the last track point to the destination airport.
        extra_end = haversine(
            df.iloc[-1]["lat"], df.iloc[-1]["lng"], flight["ADESLat"], flight["ADESLong"]
        )
        total_distance += extra_end
    return total_distance


def safe_parse_datetime(dt_string):
    """Parse a datetime string and handle excessive fractional seconds."""
    try:
        return datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(dt_string[:26], "%Y-%m-%d %H:%M:%S.%f")


def calculate_phase_distance(base_df, flight_id, df, threshold=15):
    """Calculate distances for different flight phases."""
    flight = get_flight_series(base_df, flight_id)
    df["timestamp"] = pd.to_datetime(
        df["dateReference"].astype(str) + " " + df["time"].astype(str), format="%Y-%m-%d %H:%M:%S"
    )
    df = bound_trace(flight["ATOT_track"], flight["ALDT_track"], df)
    phases_distances = {}

    phases = [
        (flight["dtPhaseStart_CLIMB"], flight["dtPhaseEnd_CLIMB"], "CLIMB"),
        (flight["dtPhaseStart_CRUISE"], flight["dtPhaseEnd_CRUISE"], "CRUISE"),
        (flight["dtPhaseStart_DESCENT"], flight["dtPhaseEnd_DESCENT"], "DESCENT"),
    ]

    for phase_start, phase_end, phase_label in phases:
        phase_start_dt = safe_parse_datetime(str(phase_start))
        phase_end_dt = safe_parse_datetime(str(phase_end))
        phase_df = df[
            (df["timestamp"] >= phase_start_dt) & (df["timestamp"] <= phase_end_dt)
        ].sort_values("timestamp")
        extra_segment = 0
        internal_distance = 0
        if not phase_df.empty:
            # gap at the beginning:
            if phase_label == "CLIMB":
                first_point_time = df.iloc[0]["timestamp"]
                gap_start = abs((first_point_time - flight["ATOT_track"]).total_seconds() / 60)
                if gap_start > threshold:
                    # distance from departure airport to the first track point.
                    extra_segment = haversine(
                        flight["ADEPLat"], flight["ADEPLong"], df.iloc[0]["lat"], df.iloc[0]["lng"]
                    )

            # gap at the end:
            if phase_label == "DESCENT":
                last_point_time = df.iloc[-1]["timestamp"]
                gap_end = (flight["ALDT_track"] - last_point_time).total_seconds()
                if gap_end > threshold:
                    extra_segment = haversine(
                        df.iloc[-1]["lat"],
                        df.iloc[-1]["lng"],
                        flight["ADESLat"],
                        flight["ADESLong"],
                    )

            # internal distance for the phase
            lat = phase_df["lat"].values
            lon = phase_df["lng"].values
            lat_next = np.append(lat[1:], np.nan)
            lon_next = np.append(lon[1:], np.nan)
            phase_df["segment_distance"] = vectorized_haversine(lat, lon, lat_next, lon_next)
            internal_distance = phase_df["segment_distance"].sum()

        total_phase_distance = extra_segment + internal_distance + extra_segment
        phases_distances[phase_label + "_distance"] = total_phase_distance

    return phases_distances


def calculate_phase_durations(df):
    """Calculate duration of flight phases in hours and fill missing values with median."""
    phase_pairs = [
        ("dtPhaseStart_CLIMB", "dtPhaseEnd_CLIMB"),
        ("dtPhaseStart_CRUISE", "dtPhaseEnd_CRUISE"),
        ("dtPhaseStart_DESCENT", "dtPhaseEnd_DESCENT"),
    ]

    df = df.copy()
    for start, end in phase_pairs:
        phase = start.replace("dtPhaseStart_", "")
        duration = (df[end] - df[start]).dt.total_seconds() / 3600  # hours
        df[f"{phase}_duration"] = duration.where(
            df[start].notna() & df[end].notna()
        )  # compute duration only for phases with start and end
    return df


def calculate_flight_duration(df):
    """Calculate total flight duration in hours."""
    df = df.copy()
    df["flight_duration"] = (
        df["ALDT_track"] - df["ATOT_track"]
    ).dt.total_seconds() / 3600  # hours
    return df


def extract_hour_day(df, col_name):
    """Extract hour and day of year from a datetime column."""
    df[f"{col_name}_hour"] = df[col_name].dt.hour
    df[f"{col_name}_day"] = df[col_name].dt.dayofyear
    return df


def connect_to_db():
    """Establish a connection to the database using secure credentials."""
    try:
        host = os.getenv("DB_HOST", r"10.232.0.145\ZXVW0104,1433")
        database = os.getenv("DB_NAME", "DWH")
        uid = os.getenv("DB_USER", "jaguevara")
        pwd = os.getenv("DB_PASS", getpass.getpass("Enter database password: "))
        conn_str = f"Driver=ODBC Driver 17 for SQL Server;Server={host};Database={database};UID={uid};PWD={pwd}"
        return pyodbc.connect(conn_str)
    except pyodbc.Error as ex:
        print("Database error:", ex)
    except Exception as e:
        print("An error occurred:", e)


def execute_query(cnxn, flight_id, sample_rate=5):
    """Execute SQL query to retrieve flight track data at a specified sampling rate."""
    try:
        sql = f"""
            WITH SampledData AS (
                SELECT 
                    a.adep, a.ades, b.flightKey, b.dateReference, c.time, b.lat, b.lng, b.modo_c, b.vel_mod, b.vel_z,
                    ROW_NUMBER() OVER (ORDER BY b.dateReference, c.time) AS rn,
                    LEAD(b.dateReference) OVER (ORDER BY b.dateReference, c.time) AS next_dateReference
                FROM dwh.dbo.dimFlowsFlights a
                INNER JOIN dwh.dbo.flowsTracksFacts b ON a.flightKey = b.flightKey
                INNER JOIN dwh.dbo.CalendarTime c ON c.timeKey = b.time
                WHERE a.flightKey = {flight_id} 
                AND b.principal = 1
            )
            SELECT 
                adep, ades, flightKey, dateReference, time, lat, lng, modo_c, vel_mod, vel_z
            FROM SampledData
            WHERE rn % {sample_rate} = 0;
        """
        query_df = pd.read_sql(sql, cnxn)
        return query_df.sort_values("time")

    except pyodbc.Error as ex:
        print("Database error:", ex)
    except Exception as e:
        print("An error occurred:", e)


def get_flight_profile(base_df, flight_id, df):
    """
    Compute flight phase statistics for modo_c, vel_mod, vel_z, and delta_modo_c
    for CLIMB, CRUISE, and DESCENT phases.

    Returns a dictionary with:
        - median and variance of modo_c, vel_mod, vel_z
        - delta_modo_c: difference between last and first value
    """
    flight = get_flight_series(base_df, flight_id)

    # Combine date and time into a proper timestamp
    df["timestamp"] = pd.to_datetime(
        df["dateReference"].astype(str) + " " + df["time"].astype(str), format="%Y-%m-%d %H:%M:%S"
    )

    # Filter data to only include between ATOT and ALDT
    df = bound_trace(flight["ATOT_track"], flight["ALDT_track"], df)

    # Define flight phases with their start/end timestamps and labels
    phase_definitions = [
        (flight["dtPhaseStart_CLIMB"], flight["dtPhaseEnd_CLIMB"], "CLIMB"),
        (flight["dtPhaseStart_CRUISE"], flight["dtPhaseEnd_CRUISE"], "CRUISE"),
        (flight["dtPhaseStart_DESCENT"], flight["dtPhaseEnd_DESCENT"], "DESCENT"),
    ]

    # Features to compute statistics for
    features = ["modo_c", "vel_mod", "vel_z"]
    stats = ["median", "variance"]

    profile = {}

    for start, end, label in phase_definitions:
        start_dt = safe_parse_datetime(str(start))
        end_dt = safe_parse_datetime(str(end))

        # Slice the dataframe to the phase
        phase_df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].sort_values(
            "timestamp"
        )

        if not phase_df.empty:
            # Compute median and variance for each feature
            for feature in features:
                profile[f"{label}_{feature}_median"] = phase_df[feature].median(skipna=True)
                profile[f"{label}_{feature}_variance"] = phase_df[feature].var(skipna=True)

            # Compute delta_modo_c
            profile[f"{label}_delta_modo_c"] = (
                phase_df.iloc[-1]["modo_c"] - phase_df.iloc[0]["modo_c"]
            )
        else:
            # Assign NaN if no data for the phase
            for feature in features:
                profile[f"{label}_{feature}_median"] = np.nan
                profile[f"{label}_{feature}_variance"] = np.nan
            profile[f"{label}_delta_modo_c"] = np.nan

    return profile


def get_initial_vel(base_df, flight_id, df):
    """
    Computes mean and variance of vel_mod and vel_z during specific FL intervals within the CLIMB phase.
    Returns a dictionary of statistics labeled by FL intervals.
    """
    flight = get_flight_series(base_df, flight_id)

    # Prepare timestamp column
    df["timestamp"] = pd.to_datetime(
        df["dateReference"].astype(str) + " " + df["time"].astype(str), format="%Y-%m-%d %H:%M:%S"
    )

    # Filter to CLIMB phase trace
    df = bound_trace(flight["dtPhaseStart_CLIMB"], flight["dtPhaseEnd_CLIMB"], df)

    # Define FL intervals and corresponding labels
    intervals = [
        (0, 30, "0-30fl"),
        (0, 60, "0-60fl"),
        (0, 90, "0-90fl"),
        (30, 60, "30-60fl"),
        (30, 90, "30-90fl"),
        (60, 90, "60-90fl"),
    ]

    vel_dict = {}

    for start_fl, end_fl, label in intervals:
        try:
            # Get data slice within the interval
            interval_df = df[(df["modo_c"] >= start_fl) & (df["modo_c"] <= end_fl)].sort_values(
                "timestamp"
            )

            if interval_df.empty:
                stats = get_empty_vel_stats(label)
            else:
                stats = compute_velocity_stats(interval_df, label)

            vel_dict.update(stats)

        except Exception:
            # Handle cases where index access (e.g., iloc[-1]) fails
            vel_dict.update(get_empty_vel_stats(label, partial=True))

    return vel_dict


def compute_velocity_stats(df, label):
    """Compute mean and variance for vel_mod and vel_z."""
    return {
        f"CLIMB_vel_mod_mean_{label}": df["vel_mod"].mean(skipna=True),
        f"CLIMB_vel_mod_variance_{label}": df["vel_mod"].var(skipna=True),
        f"CLIMB_vel_z_mean_{label}": df["vel_z"].mean(skipna=True),
        f"CLIMB_vel_z_variance_{label}": df["vel_z"].var(skipna=True),
    }


def get_empty_vel_stats(label, partial=False):
    """Return NaNs for missing or failed intervals. If partial=True, exclude variance keys."""
    stats = {f"CLIMB_vel_mod_mean_{label}": np.nan, f"CLIMB_vel_z_mean_{label}": np.nan}
    if not partial:
        stats.update(
            {f"CLIMB_vel_mod_variance_{label}": np.nan, f"CLIMB_vel_z_variance_{label}": np.nan}
        )
    return stats


# Split dataset into Model 1 (flights with adep in Spain) and Model 2 (rest of the flights)
# --- Constants for Configuration ---
# Airport prefixes for Model 1 (e.g., Spain)
SPAIN_AIRPORT_PREFIXES: tuple[str, str] = ("LE", "GC")

# List of columns to drop, excluding flightKey initially
COMMON_COLUMNS_TO_DROP_EXCEPT_FK = [
    # indices
    "CFMUflightKey",
    "callsign",
    # aircraft and aerodrome irrelevant features
    "aircraftOperator",
    "aircraftNumber",
    "previousAdes",
    "EngineType",
    "operatingAircraftOperator",
    "adep",  # Dropping adep and ades as per original code
    "ades",
    "aircraftRegistration",
    "altAdes",
    "arrivalRunway",
    "departureRunway",
    "Description",
    # irrelevant trace feature (commented out as it was in original)
    # dates (temporal features potentially derived elsewhere or not needed)
    "ATOT_track",
    "ALDT_track",
    "dtPhaseStart_CLIMB",
    "dtPhaseEnd_CLIMB",
    "dtPhaseStart_CRUISE",
    "dtPhaseEnd_CRUISE",
    "dtPhaseStart_DESCENT",
    "dtPhaseEnd_DESCENT",
]

# Dictionary for renaming columns (replacing '-' with '_')
# This can potentially be generated programmatically if the pattern is consistent
RENAME_COLUMNS_MAP = {
    "CLIMB_vel_mod_variance_0-30fl": "CLIMB_vel_mod_variance_0_30fl",
    "CLIMB_vel_z_variance_0-30fl": "CLIMB_vel_z_variance_0_30fl",
    "CLIMB_vel_mod_variance_0-60fl": "CLIMB_vel_mod_variance_0_60fl",
    "CLIMB_vel_z_variance_0-60fl": "CLIMB_vel_z_variance_0_60fl",
    "CLIMB_vel_mod_variance_0-90fl": "CLIMB_vel_mod_variance_0_90fl",
    "CLIMB_vel_z_variance_0-90fl": "CLIMB_vel_z_variance_0_90fl",
    "CLIMB_vel_mod_variance_30-60fl": "CLIMB_vel_mod_variance_30_60fl",
    "CLIMB_vel_z_variance_30-60fl": "CLIMB_vel_z_variance_30_60fl",
    "CLIMB_vel_mod_variance_30-90fl": "CLIMB_vel_mod_variance_30_90fl",
    "CLIMB_vel_z_variance_30-90fl": "CLIMB_vel_z_variance_30_90fl",
    "CLIMB_vel_mod_variance_60-90fl": "CLIMB_vel_mod_variance_60_90fl",
    "CLIMB_vel_z_variance_60-90fl": "CLIMB_vel_z_variance_60_90fl",
    "CLIMB_vel_mod_mean_0-30fl": "CLIMB_vel_mod_mean_0_30fl",
    "CLIMB_vel_z_mean_0-30fl": "CLIMB_vel_z_mean_0_30fl",
    "CLIMB_vel_mod_mean_0-60fl": "CLIMB_vel_mod_mean_0_60fl",
    "CLIMB_vel_z_mean_0-60fl": "CLIMB_vel_z_mean_0_60fl",
    "CLIMB_vel_mod_mean_0-90fl": "CLIMB_vel_mod_mean_0_90fl",
    "CLIMB_vel_z_mean_0-90fl": "CLIMB_vel_z_mean_0_90fl",
    "CLIMB_vel_mod_mean_30-60fl": "CLIMB_vel_mod_mean_30_60fl",
    "CLIMB_vel_z_mean_30-60fl": "CLIMB_vel_z_mean_30_60fl",
    "CLIMB_vel_mod_mean_30-90fl": "CLIMB_vel_mod_mean_30_90fl",
    "CLIMB_vel_z_mean_30-90fl": "CLIMB_vel_z_mean_30_90fl",
    "CLIMB_vel_mod_mean_60-90fl": "CLIMB_vel_mod_mean_60_90fl",
    "CLIMB_vel_z_mean_60-90fl": "CLIMB_vel_z_mean_60_90fl",
}

# --- Core Processing Function ---


def _process_dataframe_core(data, is_spain_origin, remove_columns_model_2, keep_flightkey=True):

    # 1. Filter data based on origin
    if is_spain_origin == "m1":
        processed_data = data[data["adep"].str.startswith(SPAIN_AIRPORT_PREFIXES)].copy()
        print(f"Filtered for flights from Spain ({SPAIN_AIRPORT_PREFIXES}).")
    elif is_spain_origin == "m2":
        processed_data = data[~data["adep"].str.startswith(SPAIN_AIRPORT_PREFIXES)].copy()
        print(f"Filtered for flights NOT from Spain ({SPAIN_AIRPORT_PREFIXES}).")
    elif is_spain_origin == "H":
        processed_data = data[data["wake"] == "H"].copy()
        print(f"Filtered for flights with Heavy wake category.")
    elif is_spain_origin == "M":
        processed_data = data[data["wake"] == "M"].copy()
        print(f"Filtered for flights with Medium wake category.")

    # 2. Determine columns to drop
    columns_to_drop_final = COMMON_COLUMNS_TO_DROP_EXCEPT_FK.copy()
    if not keep_flightkey:
        # Ensure flightKey is in the original data before trying to drop
        if "flightKey" in processed_data.columns:
            columns_to_drop_final.append("flightKey")
        else:
            print("Warning: 'flightKey' not found in DataFrame, cannot drop.")

    # Filter out columns from the drop list that don't exist in the DataFrame
    existing_columns_to_drop = [
        col for col in columns_to_drop_final if col in processed_data.columns
    ]

    # 3. Drop specified columns
    if existing_columns_to_drop:
        processed_data = processed_data.drop(columns=existing_columns_to_drop)
    # 4. Rename columns
    # Filter the rename map to only include columns that still exist after dropping
    existing_rename_map = {
        old_name: new_name
        for old_name, new_name in RENAME_COLUMNS_MAP.items()
        if old_name in processed_data.columns
    }

    if existing_rename_map:
        processed_data = processed_data.rename(columns=existing_rename_map)
        # print(f"Renamed columns: {list(existing_rename_map.keys())}") # Optional: print renamed columns

    # 5. Remove columns for model 2. This columns are specific climb features that will not be considered in model 2
    if remove_columns_model_2:
        processed_data = processed_data.drop(columns=existing_rename_map.values())

    # 5. Handle missing values (drop rows with any remaining NaNs)
    rows_before_dropna = len(processed_data)
    processed_data = processed_data.dropna()
    rows_dropped_nan = rows_before_dropna - len(processed_data)

    if rows_dropped_nan > 0:
        print(f"Rows with NaN features dropped: {rows_dropped_nan}")
    else:
        print("No rows with NaN features needed to be dropped.")

    return processed_data


# --- Wrapper Functions (using the core function) ---


def process_data_model_1(data):
    print("\n--- Processing Data for Model 1 (Spain, dropping flightKey) ---")
    return _process_dataframe_core(
        data, is_spain_origin="m1", remove_columns_model_2=False, keep_flightkey=False
    )


def process_data_model_2(data):
    print("\n--- Processing Data for Model 2 (Non-Spain, dropping flightKey) ---")
    # Corrected: Removed the second drop of renamed columns.
    return _process_dataframe_core(
        data, is_spain_origin="m2", remove_columns_model_2=True, keep_flightkey=False
    )


def process_data_H(data):
    print("\n--- Processing Data for Model H (All flights, dropping Medium wake category) ---")
    return _process_dataframe_core(
        data, is_spain_origin="H", remove_columns_model_2=True, keep_flightkey=False
    )


def process_data_M(data):
    print("\n--- Processing Data for Model M (All flights, dropping Heavy wake category) ---")
    return _process_dataframe_core(
        data, is_spain_origin="M", remove_columns_model_2=True, keep_flightkey=False
    )


def preprocess_xgboost(data):
    X = data.drop(["tow"], axis=1)
    y = data["tow"]
    categorical_variables = [
        "routeType",
        "flightType",
        "wake",
        "RECATwake",
        "numberOfEngines",
        "aircraftType",
        "airlineCode",
    ]
    new_categorical_variables = []
    for col in categorical_variables:
        if col in X.columns:
            new_categorical_variables.append(col)

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.difference(
        new_categorical_variables
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(numeric_cols)),
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="error"),
                new_categorical_variables,
            ),
        ]
    )
    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def preprocess_lightgbm(data):
    X = data.drop(["tow"], axis=1)
    y = data["tow"]
    categorical_variables = [
        "routeType",
        "flightType",
        "wake",
        "RECATwake",
        "numberOfEngines",
        "aircraftType",
        "airlineCode",
    ]
    for col in categorical_variables:
        if col in X.columns:
            X[col] = X[col].astype("category")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def preprocess_catboost(data):
    if "numberOfEngines" in data.columns:
        data["numberOfEngines"] = data["numberOfEngines"].astype("int")
    X = data.drop(["tow"], axis=1)
    y = data["tow"]
    categorical_variables = [
        "routeType",
        "flightType",
        "wake",
        "RECATwake",
        "numberOfEngines",
        "aircraftType",
        "airlineCode",
    ]
    new_categorical_variables = []
    for col in categorical_variables:
        if col in X.columns:
            new_categorical_variables.append(col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_pool = Pool(X_train, y_train, cat_features=new_categorical_variables)
    test_pool = Pool(X_test, y_test, cat_features=new_categorical_variables)
    return X_train, X_test, y_train, y_test, train_pool, test_pool


def preprocess_h2o(data):
    X = data.drop(["tow"], axis=1)
    y = data["tow"]
    categorical_variables = [
        "routeType",
        "flightType",
        "wake",
        "RECATwake",
        "numberOfEngines",
        "aircraftType",
        "airlineCode",
    ]
    new_categorical_variables = []
    for col in categorical_variables:
        if col in X.columns:
            X[col] = X[col].astype("category")
            new_categorical_variables.append(col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train = h2o.H2OFrame(train_df)
    test = h2o.H2OFrame(test_df)

    return train, test, X_test, y_test


def fill_mtow_rates(df, aircraftType):
    relevant_df = df[df["aircraftType"] == aircraftType].copy()
    if relevant_df["mtow_rates"].dropna().empty:
        print(
            f"No existing 'mtow_rates' for aircraft type: {aircraftType}. Cannot fill missing values."
        )
        return df

    most_common_mtow = relevant_df["mtow_rates"].mode().iloc[0]
    lookup_cols = ["aircraftType", "airlineCode", "adep", "ades"]
    similar_flights_lookup = relevant_df[
        (relevant_df["mtow_rates"] == most_common_mtow) & (relevant_df["mtow_rates"].notna())
    ][lookup_cols + ["mtow_rates"]].drop_duplicates()
    df = pd.merge(
        df, similar_flights_lookup, how="left", on=lookup_cols, suffixes=("_original", "_lookup")
    )

    df["mtow_rates"] = df["mtow_rates_original"].fillna(df["mtow_rates_lookup"])
    df = df.drop(columns=["mtow_rates_original", "mtow_rates_lookup"])
    return df


def get_model_metrics(y_pred, y_test):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test))
    return r2, mae, rmse, mape


def get_model_figs(residuals, y_pred, y_test, title1, title2, title3, path1, path2, path3):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(residuals, linewidth=2)
    plt.title(title1)
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(path1)
    distances = np.abs(y_pred - y_test)
    norm = Normalize(vmin=0, vmax=np.max(distances))
    colors = plt.cm.coolwarm(norm(distances))  # colormap
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(x=y_pred, y=y_test, c=distances, cmap="coolwarm", marker=".", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "g-", lw=2, label="Ideal")
    plt.xlabel("Predicted tow [kg]")
    plt.ylabel("Actual tow [kg]")
    plt.title(title2)
    plt.grid(True)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Distance from Regression Line")
    plt.tight_layout()
    plt.savefig(path2)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color="green", linestyle="--", linewidth=2)
    plt.xlabel("Predicted tow [kg]")
    plt.ylabel("Residuals")
    plt.title(title3)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path3)
    plt.close()


def generate_run_name(model_name):
    """Generates a unique and descriptive run name for a model."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}"


def log_model_mlflow(
    X_data,
    y_data,
    dataset,
    model,
    lib,
    model_type,
    wake_category,
    run_name,
    X_test,
    y_test,
    test_split_ratio,
):
    print(f"Starting MLflow run: {run_name}")
    with mlflow.start_run():
        # --- 1. Log Inputs and Basic Info ---
        try:
            mlflow.log_input(dataset)
        except Exception as e:
            print(f"Warning: Could not log input dataset. Error: {e}")

        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.log_param("Train_test_split", test_split_ratio)

        params_to_log = {}
        # --- 2. Log Model Parameters ---
        if hasattr(model, "get_params") and callable(model.get_params):
            params_to_log = model.get_params()
        elif hasattr(model, "params"):  # Fallback for native LightGBM
            params_to_log = model.params
        else:
            raise Exception("Model parameters could not be retrieved.")

        mlflow.log_params(params_to_log)

        # --- 3. Model Signature and Logging ---
        signature = None
        try:
            signature = mlflow.models.infer_signature(X_data, y_data)
        except Exception as e:
            print(
                f"Warning: Could not infer model signature. Error: {e}. Proceeding without signature."
            )

        model_artifact_name = f"{run_name.replace(' ', '_')}"

        log_model_func = MODEL_LOGGERS.get(lib)
        if log_model_func:
            try:
                log_model_func(model, model_artifact_name, signature=signature)
                print(f"Logged model '{model_artifact_name}' using '{lib}' specific logger.")
            except Exception as e:
                print(f"Error logging model with {lib} specific logger: {e}")
        else:
            print(f"Warning: Specific model logging for library '{lib}' is not configured.")

        # --- 4. Predictions and Metrics ---
        y_pred = None
        try:
            if lib == "h2o":
                y_pred = model.predict(X_test).as_data_frame().values.flatten()
                y_test = X_test["tow"].as_data_frame().values.flatten()
            else:
                y_pred = model.predict(X_test)
        except Exception as e:
            print(f"Error during model prediction: {e}")
            mlflow.log_metric("prediction_status", 0)  # Indicate failure
            return

        if y_pred is not None:
            try:
                r2, mae, rmse, mape = get_model_metrics(y_pred, y_test)

                print(f"\nMetrics for run: {run_name}")
                print(f"  R^2 Score: {r2:.4f}")
                print(f"  Mean Absolute Error: {mae:.2f}")
                print(f"  Root Mean Squared Error: {rmse:.2f}")
                print(f"  Mean Absolute Percentage Error: {mape:.4f}")

                metrics_to_log = {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}
                mlflow.log_metrics(metrics_to_log)
            except Exception as e:
                print(f"Error calculating or logging metrics: {e}")
        else:
            print("Skipping metrics calculation as predictions failed or were not generated.")

        # --- 5. Log Artifacts (Figures) ---
        if model_type == "m1+m2" and wake_category == "H_wake":
            specific_folder = "H"
        elif model_type == "m1+m2" and wake_category == "M_wake":
            specific_folder = "M"
        else:
            specific_folder = f"{model_type}/{wake_category}"

        # Figure base name uses run_name to ensure uniqueness per run
        figure_file_basename = run_name.replace(" ", "_")

        source_figure_directory = os.path.join(FIGURES_DIR, specific_folder)

        figures_to_log_info = {
            FIGURE_RESIDUALS_DIST: "Residuals Distribution Plot",
            FIGURE_SCATTER_RESIDUALS: "Scatter Residuals Plot",
            FIGURE_SCATTER: "Scatter Plot",
        }

        print(
            f"\nAttempting to log figures from directory: {os.path.abspath(source_figure_directory)}"
        )
        if os.path.exists(
            os.path.join(FIGURES_DIR, specific_folder, f"{run_name}{FIGURE_FEATURE_IMPORTANCE}")
        ):
            mlflow.log_artifact(
                os.path.join(
                    FIGURES_DIR,
                    specific_folder,
                    f"{run_name}{FIGURE_FEATURE_IMPORTANCE}",
                ),
                artifact_path=ARTIFACT_SUBFOLDER_FIGURES,
            )
            print(f"  Successfully logged feature importance figure.")

        if os.path.exists(
            os.path.join(FIGURES_DIR, specific_folder, f"{run_name}{FIGURE_PARETO}")
        ):
            mlflow.log_artifact(
                os.path.join(
                    FIGURES_DIR,
                    specific_folder,
                    f"{run_name}{FIGURE_PARETO}",
                ),
                artifact_path=ARTIFACT_SUBFOLDER_FIGURES,
            )
            print(f"  Successfully logged pareto figure.")

        if os.path.exists(
            os.path.join(FIGURES_DIR, specific_folder, f"{run_name}{CSV_LEADERBOARD}")
        ):
            mlflow.log_artifact(
                os.path.join(
                    FIGURES_DIR,
                    specific_folder,
                    f"{run_name}{CSV_LEADERBOARD}",
                ),
                artifact_path=ARTIFACT_SUBFOLDER_LEADERBOARD,
            )
            print(f"  Successfully logged Leaderboard DataFrame.")

        for fig_suffix, desc in figures_to_log_info.items():
            fig_path = os.path.join(source_figure_directory, f"{figure_file_basename}{fig_suffix}")

            if os.path.exists(fig_path):
                try:
                    mlflow.log_artifact(fig_path, artifact_path=ARTIFACT_SUBFOLDER_FIGURES)
                    print(f"  Successfully logged artifact: {desc} (from {fig_path})")
                except Exception as e:
                    print(f"  Warning: Could not log artifact {fig_path}. Error: {e}")
            else:
                print(f"  Warning: Artifact file not found, skipping: {fig_path}")

    print(f"MLflow run '{run_name}' completed and logged.\n" + "=" * 40)
