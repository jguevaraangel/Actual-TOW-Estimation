import pandas as pd
import pyodbc
import seaborn as sns
import getpass
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
from catboost import CatBoostRegressor, Pool
import warnings

# ignore specific warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable"
)

# earth radius in nautical miles
EARTH_RADIUS_NM = 3440.065


def fill_with_median(series):
    """fill nan values in a series with the median of non-nan values."""
    median_value = series.median(skipna=True)
    return series.fillna(median_value)


def extract_hour_day(df, col_name):
    """extract hour and day of year from a datetime column."""
    df[f"{col_name}_hour"] = df[col_name].dt.hour
    df[f"{col_name}_day"] = df[col_name].dt.dayofyear
    return df


def calculate_phase_durations(df):
    """calculate duration of flight phases in hours and fill missing values with median."""
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

    # cols = ['CLIMB_duration', 'DESCENT_duration']
    # for col in cols:
    #     df[col] = df.groupby('wake')[col].transform(fill_with_median)

    # df['CRUISE_duration'] = df['CRUISE_duration'].fillna(df['flight_duration'] - df['CLIMB_duration'] - df['DESCENT_duration'])
    return df


def calculate_flight_duration(df):
    """calculate total flight duration in hours."""
    df = df.copy()
    df["flight_duration"] = (
        df["ALDT_track"] - df["ATOT_track"]
    ).dt.total_seconds() / 3600  # hours
    return df


def connect_to_db():
    """establish a connection to the database using secure credentials."""
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
    """execute SQL query to retrieve flight track data at a specified sampling rate."""
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


def haversine(lat1, lon1, lat2, lon2):
    """compute great-circle distance using the Haversine formula."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_NM * atan2(sqrt(a), sqrt(1 - a))


def vectorized_haversine(lat, lon, lat_next, lon_next):
    """vectorized implementation of the Haversine formula."""
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
    """bound the trace by ATOT and ALDT"""
    return df[(df["timestamp"] >= takeoff_time) & (df["timestamp"] <= landing_time)].sort_values(
        "timestamp"
    )


def get_flight_series(df, id):
    """convert df row to series"""
    return df[df["flightKey"] == id].squeeze(axis=0)


def calculate_total_distance(base_df, flight_id, df, threshold=15):
    """compute the total flight distance based on trace points."""
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
    """parse a datetime string and handle excessive fractional seconds."""
    try:
        return datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(dt_string[:26], "%Y-%m-%d %H:%M:%S.%f")


def calculate_phase_distance(base_df, flight_id, df, threshold=15):
    """calculate distances for different flight phases."""
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
        # phase_start_dt = datetime.strptime(str(phase_start), "%Y-%m-%d %H:%M:%S")
        # phase_end_dt = datetime.strptime(str(phase_end), "%Y-%m-%d %H:%M:%S")
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


def get_flight_profile(base_df, flight_id, df):
    """get flight profile, median and variance of:
     - modo_c
     - vel_mod
     - vel_z
    and:
     - modo_c_last - modo_c_first
    """
    flight = get_flight_series(base_df, flight_id)
    df["timestamp"] = pd.to_datetime(
        df["dateReference"].astype(str) + " " + df["time"].astype(str), format="%Y-%m-%d %H:%M:%S"
    )
    df = bound_trace(flight["ATOT_track"], flight["ALDT_track"], df)
    phases_dict = {}
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

        if not phase_df.empty:
            phases_dict[phase_label + "_modo_c_median"] = phase_df["modo_c"].median(skipna=True)
            phases_dict[phase_label + "_modo_c_variance"] = phase_df["modo_c"].var(skipna=True)

            phases_dict[phase_label + "_vel_mod_median"] = phase_df["vel_mod"].median(skipna=True)
            phases_dict[phase_label + "_vel_mod_variance"] = phase_df["vel_mod"].var(skipna=True)

            phases_dict[phase_label + "_vel_z_median"] = phase_df["vel_z"].median(skipna=True)
            phases_dict[phase_label + "_vel_z_variance"] = phase_df["vel_z"].var(skipna=True)

            phases_dict[phase_label + "_delta_modo_c"] = (
                phase_df.iloc[-1]["modo_c"] - phase_df.iloc[0]["modo_c"]
            )

        else:
            phases_dict[phase_label + "_modo_c_median"] = np.nan
            phases_dict[phase_label + "_modo_c_variance"] = np.nan

            phases_dict[phase_label + "_vel_mod_median"] = np.nan
            phases_dict[phase_label + "_vel_mod_variance"] = np.nan

            phases_dict[phase_label + "_vel_z_median"] = np.nan
            phases_dict[phase_label + "_vel_z_variance"] = np.nan

            phases_dict[phase_label + "_delta_modo_c"] = np.nan

    return phases_dict


def get_initial_vel(base_df, flight_id, df):
    """ """
    flight = get_flight_series(base_df, flight_id)
    df["timestamp"] = pd.to_datetime(
        df["dateReference"].astype(str) + " " + df["time"].astype(str), format="%Y-%m-%d %H:%M:%S"
    )
    df = bound_trace(flight["dtPhaseStart_CLIMB"], flight["dtPhaseEnd_CLIMB"], df)
    vel_dict = {}

    # All possible height intervals
    vel_intervals = [
        (0, 30, "0-30fl"),
        (0, 60, "0-60fl"),
        (0, 90, "0-90fl"),
        (30, 60, "30-60fl"),
        (30, 90, "30-90fl"),
        (60, 90, "60-90fl"),
    ]

    for start_height, end_height, label in vel_intervals:
        try:
            start_dt = safe_parse_datetime(str(df.iloc[0]["timestamp"]))
            end_dt = safe_parse_datetime(
                str(
                    df[(df["modo_c"] >= start_height) & (df["modo_c"] <= end_height)].iloc[-1][
                        "timestamp"
                    ]
                )
            )

            slice = df[
                (df["timestamp"] >= start_dt)
                & (df["timestamp"] <= end_dt)
                & (df["modo_c"] >= start_height)
                & (df["modo_c"] <= end_height)
            ].sort_values("timestamp")

            if not slice.empty:
                vel_dict[f"CLIMB_vel_mod_mean_{label}"] = slice["vel_mod"].mean(skipna=True)
                vel_dict[f"CLIMB_vel_mod_variance_{label}"] = slice["vel_mod"].var(skipna=True)

                vel_dict[f"CLIMB_vel_z_mean_{label}"] = slice["vel_z"].mean(skipna=True)
                vel_dict[f"CLIMB_vel_z_variance_{label}"] = slice["vel_z"].var(skipna=True)

            else:
                vel_dict[f"CLIMB_vel_mod_mean_{label}"] = np.nan
                vel_dict[f"CLIMB_vel_mod_variance_{label}"] = np.nan

                vel_dict[f"CLIMB_vel_z_mean_{label}"] = np.nan
                vel_dict[f"CLIMB_vel_z_variance_{label}"] = np.nan
        except Exception as e:
            vel_dict[f"CLIMB_vel_mod_mean_{label}"] = np.nan
            vel_dict[f"CLIMB_vel_z_mean_{label}"] = np.nan

    return vel_dict


def process_data(data):
    columns_to_drop = [
        "flightKey",
        "CFMUflightKey",
        "callsign",
        "aircraftOperator",
        "aircraftNumber",
        "previousAdes",
        "EngineType",
        "operatingAircraftOperator",
        "adep",
        "ades",
        "aircraftRegistration",
        "altAdes",
        "arrivalRunway",
        "departureRunway",
        "Description",
        "ATOT_track",
        "ALDT_track",
        "dtPhaseStart_CLIMB",
        "dtPhaseEnd_CLIMB",
        "dtPhaseStart_CRUISE",
        "dtPhaseEnd_CRUISE",
        "dtPhaseStart_DESCENT",
        "dtPhaseEnd_DESCENT",
    ]

    # Drop columns with missing values
    data = data.drop(columns=columns_to_drop)
    data = data.rename(
        columns={
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
    )
    data = data.drop(
        columns=[
            "CLIMB_vel_mod_variance_0_30fl",
            "CLIMB_vel_z_variance_0_30fl",
            "CLIMB_vel_mod_variance_0_60fl",
            "CLIMB_vel_z_variance_0_60fl",
            "CLIMB_vel_mod_variance_0_90fl",
            "CLIMB_vel_z_variance_0_90fl",
            "CLIMB_vel_mod_variance_30_60fl",
            "CLIMB_vel_z_variance_30_60fl",
            "CLIMB_vel_mod_variance_30_90fl",
            "CLIMB_vel_z_variance_30_90fl",
            "CLIMB_vel_mod_variance_60_90fl",
            "CLIMB_vel_z_variance_60_90fl",
            "CLIMB_vel_mod_mean_0_30fl",
            "CLIMB_vel_z_mean_0_30fl",
            "CLIMB_vel_mod_mean_0_60fl",
            "CLIMB_vel_z_mean_0_60fl",
            "CLIMB_vel_mod_mean_0_90fl",
            "CLIMB_vel_z_mean_0_90fl",
            "CLIMB_vel_mod_mean_30_60fl",
            "CLIMB_vel_z_mean_30_60fl",
            "CLIMB_vel_mod_mean_30_90fl",
            "CLIMB_vel_z_mean_30_90fl",
            "CLIMB_vel_mod_mean_60_90fl",
            "CLIMB_vel_z_mean_60_90fl",
        ]
    )
    rows_before = len(data)
    data = data.dropna()
    print(f"Rows with nan features dropped: {rows_before - len(data)}")
    return data


def process_data_model_1(data):
    data = data[data["adep"].str.startswith("LE") | data["adep"].str.startswith("GC")]
    columns_to_drop = [
        # indices
        "flightKey",
        "CFMUflightKey",
        "callsign",
        # aircraft and aerodrome irrelevant features
        "aircraftOperator",
        "aircraftNumber",
        "previousAdes",
        "EngineType",
        "operatingAircraftOperator",
        "adep",
        "ades",
        "aircraftRegistration",
        "altAdes",
        "arrivalRunway",
        "departureRunway",
        "Description",
        # irrelevant trace feature
        # dates
        "ATOT_track",
        "ALDT_track",
        "dtPhaseStart_CLIMB",
        "dtPhaseEnd_CLIMB",
        "dtPhaseStart_CRUISE",
        "dtPhaseEnd_CRUISE",
        "dtPhaseStart_DESCENT",
        "dtPhaseEnd_DESCENT",
    ]

    # Drop columns with missing values
    rows_before = len(data)
    data = data.drop(columns=columns_to_drop)
    data = data.dropna()
    print(f"Rows with nan features dropped: {rows_before - len(data)}")

    data = data.rename(
        columns={
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
    )

    return data


def process_data_model_2(data):
    data = data[~(data["adep"].str.startswith("LE") | data["adep"].str.startswith("GC"))]
    columns_to_drop = [
        "flightKey",
        "CFMUflightKey",
        "callsign",
        "aircraftOperator",
        "aircraftNumber",
        "previousAdes",
        "EngineType",
        "operatingAircraftOperator",
        "adep",
        "ades",
        "aircraftRegistration",
        "altAdes",
        "arrivalRunway",
        "departureRunway",
        "Description",
        "ATOT_track",
        "ALDT_track",
        "dtPhaseStart_CLIMB",
        "dtPhaseEnd_CLIMB",
        "dtPhaseStart_CRUISE",
        "dtPhaseEnd_CRUISE",
        "dtPhaseStart_DESCENT",
        "dtPhaseEnd_DESCENT",
    ]

    # Drop columns with missing values
    data = data.drop(columns=columns_to_drop)
    data = data.rename(
        columns={
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
    )
    data = data.drop(
        columns=[
            "CLIMB_vel_mod_variance_0_30fl",
            "CLIMB_vel_z_variance_0_30fl",
            "CLIMB_vel_mod_variance_0_60fl",
            "CLIMB_vel_z_variance_0_60fl",
            "CLIMB_vel_mod_variance_0_90fl",
            "CLIMB_vel_z_variance_0_90fl",
            "CLIMB_vel_mod_variance_30_60fl",
            "CLIMB_vel_z_variance_30_60fl",
            "CLIMB_vel_mod_variance_30_90fl",
            "CLIMB_vel_z_variance_30_90fl",
            "CLIMB_vel_mod_variance_60_90fl",
            "CLIMB_vel_z_variance_60_90fl",
            "CLIMB_vel_mod_mean_0_30fl",
            "CLIMB_vel_z_mean_0_30fl",
            "CLIMB_vel_mod_mean_0_60fl",
            "CLIMB_vel_z_mean_0_60fl",
            "CLIMB_vel_mod_mean_0_90fl",
            "CLIMB_vel_z_mean_0_90fl",
            "CLIMB_vel_mod_mean_30_60fl",
            "CLIMB_vel_z_mean_30_60fl",
            "CLIMB_vel_mod_mean_30_90fl",
            "CLIMB_vel_z_mean_30_90fl",
            "CLIMB_vel_mod_mean_60_90fl",
            "CLIMB_vel_z_mean_60_90fl",
        ]
    )
    rows_before = len(data)
    data = data.dropna()
    print(f"Rows with nan features dropped: {rows_before - len(data)}")
    return data


def process_data_model_1_w_fk(data):
    data = data[data["adep"].str.startswith("LE") | data["adep"].str.startswith("GC")]
    columns_to_drop = [
        # indices
        "CFMUflightKey",
        "callsign",
        # aircraft and aerodrome irrelevant features
        "aircraftOperator",
        "aircraftNumber",
        "previousAdes",
        "EngineType",
        "operatingAircraftOperator",
        "adep",
        "ades",
        "aircraftRegistration",
        "altAdes",
        "arrivalRunway",
        "departureRunway",
        "Description",
        # irrelevant trace feature
        # dates
        "ATOT_track",
        "ALDT_track",
        "dtPhaseStart_CLIMB",
        "dtPhaseEnd_CLIMB",
        "dtPhaseStart_CRUISE",
        "dtPhaseEnd_CRUISE",
        "dtPhaseStart_DESCENT",
        "dtPhaseEnd_DESCENT",
    ]

    # Drop columns with missing values
    rows_before = len(data)
    data = data.drop(columns=columns_to_drop)
    data = data.dropna()
    print(f"Rows with nan features dropped: {rows_before - len(data)}")

    data = data.rename(
        columns={
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
    )

    return data


def process_data_model_2_w_fk(data):
    data = data[~(data["adep"].str.startswith("LE") | data["adep"].str.startswith("GC"))]
    columns_to_drop = [
        "CFMUflightKey",
        "callsign",
        "aircraftOperator",
        "aircraftNumber",
        "previousAdes",
        "EngineType",
        "operatingAircraftOperator",
        "adep",
        "ades",
        "aircraftRegistration",
        "altAdes",
        "arrivalRunway",
        "departureRunway",
        "Description",
        "ATOT_track",
        "ALDT_track",
        "dtPhaseStart_CLIMB",
        "dtPhaseEnd_CLIMB",
        "dtPhaseStart_CRUISE",
        "dtPhaseEnd_CRUISE",
        "dtPhaseStart_DESCENT",
        "dtPhaseEnd_DESCENT",
    ]

    # Drop columns with missing values
    data = data.drop(columns=columns_to_drop)
    data = data.rename(
        columns={
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
    )
    data = data.drop(
        columns=[
            "CLIMB_vel_mod_variance_0_30fl",
            "CLIMB_vel_z_variance_0_30fl",
            "CLIMB_vel_mod_variance_0_60fl",
            "CLIMB_vel_z_variance_0_60fl",
            "CLIMB_vel_mod_variance_0_90fl",
            "CLIMB_vel_z_variance_0_90fl",
            "CLIMB_vel_mod_variance_30_60fl",
            "CLIMB_vel_z_variance_30_60fl",
            "CLIMB_vel_mod_variance_30_90fl",
            "CLIMB_vel_z_variance_30_90fl",
            "CLIMB_vel_mod_variance_60_90fl",
            "CLIMB_vel_z_variance_60_90fl",
            "CLIMB_vel_mod_mean_0_30fl",
            "CLIMB_vel_z_mean_0_30fl",
            "CLIMB_vel_mod_mean_0_60fl",
            "CLIMB_vel_z_mean_0_60fl",
            "CLIMB_vel_mod_mean_0_90fl",
            "CLIMB_vel_z_mean_0_90fl",
            "CLIMB_vel_mod_mean_30_60fl",
            "CLIMB_vel_z_mean_30_60fl",
            "CLIMB_vel_mod_mean_30_90fl",
            "CLIMB_vel_z_mean_30_90fl",
            "CLIMB_vel_mod_mean_60_90fl",
            "CLIMB_vel_z_mean_60_90fl",
        ]
    )
    rows_before = len(data)
    data = data.dropna()
    print(f"Rows with nan features dropped: {rows_before - len(data)}")
    return data


def preprocess_xgboost(data):
    X = data.drop(["tow"], axis=1)
    y = data["tow"]

    # Identify Categorical Variables
    categorical_variables = [
        "routeType",
        "flightType",
        "wake",
        "RECATwake",
        "numberOfEngines",
        "aircraftType",
        "airlineCode",
    ]

    if "wake" not in X.columns:
        categorical_variables.remove("wake")

    # Identify numeric columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.difference(
        categorical_variables
    )

    # Preprocess all the dataset
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(numeric_cols)),
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="error"),
                categorical_variables,
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

    # Identify Categorical Variables
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
    data["numberOfEngines"] = data["numberOfEngines"].astype("int")
    X = data.drop(["tow"], axis=1)
    y = data["tow"]

    # Identify Categorical Variables
    categorical_variables = [
        "routeType",
        "flightType",
        "wake",
        "RECATwake",
        "numberOfEngines",
        "aircraftType",
        "airlineCode",
    ]

    if "wake" not in X.columns:
        categorical_variables.remove("wake")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_pool = Pool(X_train, y_train, cat_features=categorical_variables)
    test_pool = Pool(X_test, y_test, cat_features=categorical_variables)

    return X_train, X_test, y_train, y_test, train_pool, test_pool


def get_model_metrics(y_pred, X_test, y_test):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
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

    plt.show()
