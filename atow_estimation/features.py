import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm
from atow_estimation.paths import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from atow_estimation.utils import *

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


def load_dataset():
    """Load all required datasets with appropriate datatypes."""
    return pd.read_csv(os.path.join(INTERIM_DATA_DIR, "initial_data.csv"))


def load_or_process_csv(filename, processing_function):
    """Load a CSV file if it exists, otherwise process the data and save it to a CSV file."""
    file_path = os.path.join(INTERIM_DATA_DIR, filename)
    if os.path.exists(file_path):
        print(f"Loading cached file: {filename}")
        return pd.read_csv(file_path)
    else:
        print(f"Processing and creating: {filename}")
        data = processing_function()
        data.to_csv(file_path, index=False)
        return data


def process_non_empty_flights(data, cnn):
    """Remove flights without trajectory points."""
    flights_wo_trajectory = []
    for flight_id in tqdm(data["flightKey"], desc="Checking flights without trace points"):
        query_df = execute_query(cnn, flight_id)
        if query_df.empty:
            flights_wo_trajectory.append(flight_id)
    data = data[~data["flightKey"].isin(flights_wo_trajectory)]
    # data = data[~data['flightKey'].isin([18709432, 19994755, 19936534])]
    return data


def process_flight_phases_features(data, cnn):
    # datetime format
    data["ATOT_track"] = pd.to_datetime(data["ATOT_track"])
    data["ALDT_track"] = pd.to_datetime(data["ALDT_track"])

    phase_times_cols = [
        "dtPhaseStart_CLIMB",
        "dtPhaseEnd_CLIMB",
        "dtPhaseStart_CRUISE",
        "dtPhaseEnd_CRUISE",
        "dtPhaseStart_DESCENT",
        "dtPhaseEnd_DESCENT",
    ]
    for col in phase_times_cols:
        data[col] = pd.to_datetime(data[col])

    # set correct end_climb and start_descent
    data["takeoff_diff"] = abs(
        (data["dtPhaseStart_CLIMB"] - data["ATOT_track"]).dt.total_seconds() / 3600
    )
    data["landing_diff"] = abs(
        (data["ALDT_track"] - data["dtPhaseEnd_DESCENT"]).dt.total_seconds() / 3600
    )
    data["dtPhaseStart_CLIMB"] = data["dtPhaseStart_CLIMB"].mask(
        data["takeoff_diff"] > 0.5, np.nan
    )  # 30 min difference
    data["dtPhaseEnd_DESCENT"] = data["dtPhaseEnd_DESCENT"].mask(
        data["landing_diff"] > 0.5, np.nan
    )
    data["dtPhaseEnd_CLIMB"] = data["dtPhaseEnd_CLIMB"].mask(
        data["dtPhaseStart_CLIMB"].isna(), np.nan
    )
    data["dtPhaseStart_DESCENT"] = data["dtPhaseStart_DESCENT"].mask(
        data["dtPhaseEnd_DESCENT"].isna(), np.nan
    )
    data["dtPhaseStart_CLIMB"] = data["dtPhaseStart_CLIMB"].fillna(data["ATOT_track"])
    data["dtPhaseEnd_DESCENT"] = data["dtPhaseEnd_DESCENT"].fillna(data["ALDT_track"])
    # calculate again to compare
    data["takeoff_diff"] = abs(
        (data["dtPhaseStart_CLIMB"] - data["ATOT_track"]).dt.total_seconds() / 3600
    )
    data["landing_diff"] = abs(
        (data["ALDT_track"] - data["dtPhaseEnd_DESCENT"]).dt.total_seconds() / 3600
    )
    # set correct start_cruise and end_cruise
    data["dtPhaseStart_CRUISE"] = data["dtPhaseStart_CRUISE"].mask(
        data["dtPhaseEnd_CLIMB"].isna(), np.nan
    )
    data["dtPhaseEnd_CRUISE"] = data["dtPhaseEnd_CRUISE"].mask(
        data["dtPhaseStart_DESCENT"].isna(), np.nan
    )

    # order columns for visualization
    phase_times_cols = [
        "dtPhaseStart_CLIMB",
        "dtPhaseEnd_CLIMB",
        "dtPhaseStart_CRUISE",
        "dtPhaseEnd_CRUISE",
        "dtPhaseStart_DESCENT",
        "dtPhaseEnd_DESCENT",
    ]
    remaining_columns = [col for col in data.columns if col not in phase_times_cols]
    data = data[remaining_columns + phase_times_cols]

    data = extract_hour_day(data, "ATOT_track")
    data = extract_hour_day(data, "ALDT_track")
    data = calculate_flight_duration(data)
    data = calculate_phase_durations(data)

    # set end_climb and start_descent with medians of each phase
    # for CLIMB phase
    median_duration = pd.to_timedelta(
        data.groupby("wake")["CLIMB_duration"].transform("median"), unit="h"
    )
    data["dtPhaseEnd_CLIMB"] = data["dtPhaseEnd_CLIMB"].fillna(
        data["dtPhaseStart_CLIMB"] + median_duration
    )

    # for DESCENT phase
    median_duration = pd.to_timedelta(
        data.groupby("wake")["DESCENT_duration"].transform("median"), unit="h"
    )
    data["dtPhaseStart_DESCENT"] = data["dtPhaseStart_DESCENT"].fillna(
        data["dtPhaseEnd_DESCENT"] - median_duration
    )

    # calculate again phase duration for missing durations of climb and descent
    data = calculate_phase_durations(data)

    # calculate cruise duration with flight_duration and other two phases duration
    data["CRUISE_duration"] = data["CRUISE_duration"].fillna(
        data["flight_duration"] - data["CLIMB_duration"] - data["DESCENT_duration"]
    )
    data["dtPhaseStart_CRUISE"] = data["dtPhaseEnd_CLIMB"]
    data["dtPhaseEnd_CRUISE"] = data["dtPhaseStart_DESCENT"]

    # calculate distances
    data[["total_distance", "CLIMB_distance", "CRUISE_distance", "DESCENT_distance"]] = np.nan

    for flight_id in tqdm(data["flightKey"], desc="Calculating flight phases distances"):
        query_df = execute_query(cnn, flight_id)
        total_distance = calculate_total_distance(data, flight_id, query_df)
        phase_distances = calculate_phase_distance(data, flight_id, query_df)
        data.loc[
            data["flightKey"] == flight_id,
            ["total_distance", "CLIMB_distance", "CRUISE_distance", "DESCENT_distance"],
        ] = (
            total_distance,
            phase_distances.get("CLIMB_distance", np.nan),
            phase_distances.get("CRUISE_distance", np.nan),
            phase_distances.get("DESCENT_distance", np.nan),
        )

    # Median distance imputing
    for phase in ["CLIMB", "DESCENT"]:
        median_distance = data.groupby("wake")[f"{phase}_distance"].transform("median")
        overall_median = data.loc[data[f"{phase}_distance"] > 0, f"{phase}_distance"].median()
        data[f"{phase}_distance"] = data[f"{phase}_distance"].mask(
            data[f"{phase}_distance"] == 0,
            median_distance.where(median_distance > 0, overall_median),
        )

    data["haversine_dist"] = vectorized_haversine(
        data["ADEPLat"].values,
        data["ADEPLong"].values,
        data["ADESLat"].values,
        data["ADESLong"].values,
    )
    data["estimated_cruise_distance"] = (
        data["haversine_dist"] - data["CLIMB_distance"] - data["DESCENT_distance"]
    )
    data["CRUISE_distance"] = data["CRUISE_distance"].mask(
        data["CRUISE_distance"] == 0, data["estimated_cruise_distance"]
    )
    data["CRUISE_distance"] = data["CRUISE_distance"].mask(
        data["estimated_cruise_distance"] > data["CRUISE_distance"],
        data["estimated_cruise_distance"],
    )

    return data


def process_trajectory_features(data, cnn):
    """Calculate trajectory features: modo_c, vel_mod, vel_z, delta_modo_c."""
    trajectory_features = [
        "CLIMB_modo_c_median",
        "CLIMB_modo_c_variance",
        "CLIMB_vel_mod_median",
        "CLIMB_vel_mod_variance",
        "CLIMB_vel_z_median",
        "CLIMB_vel_z_variance",
        "CLIMB_delta_modo_c",
        "CRUISE_modo_c_median",
        "CRUISE_modo_c_variance",
        "CRUISE_vel_mod_median",
        "CRUISE_vel_mod_variance",
        "CRUISE_vel_z_median",
        "CRUISE_vel_z_variance",
        "CRUISE_delta_modo_c",
        "DESCENT_modo_c_median",
        "DESCENT_modo_c_variance",
        "DESCENT_vel_mod_median",
        "DESCENT_vel_mod_variance",
        "DESCENT_vel_z_median",
        "DESCENT_vel_z_variance",
        "DESCENT_delta_modo_c",
    ]

    for flight_id in tqdm(data["flightKey"], desc="Calculating trajectory features"):
        query_df = execute_query(cnn, flight_id)
        flight_profile = get_flight_profile(data, flight_id, query_df)
        values = [flight_profile.get(key, np.nan) for key in trajectory_features]
        data.loc[data["flightKey"] == flight_id, trajectory_features] = values

    for col in trajectory_features:
        median_distance = data.groupby("wake")[col].transform("median")
        data[col] = data[col].fillna(median_distance)

    return data


def process_climb_velocity(data, cnn):
    """Calculate velocity features for the early climb phase."""
    intervals = ["0-30fl", "0-60fl", "0-90fl", "30-60fl", "30-90fl", "60-90fl"]
    vel_intervals = []
    for label in intervals:
        vel_intervals.extend(
            [
                f"CLIMB_vel_mod_mean_{label}",
                f"CLIMB_vel_z_mean_{label}",
                f"CLIMB_vel_mod_variance_{label}",
                f"CLIMB_vel_z_variance_{label}",
            ]
        )

    for flight_id in tqdm(
        data["flightKey"], desc="Calculating velocity features for climb early climb phase"
    ):
        query_df = execute_query(cnn, flight_id)
        initial_vel = get_initial_vel(data, flight_id, query_df)
        values = [initial_vel.get(key, np.nan) for key in vel_intervals]
        data.loc[data["flightKey"] == flight_id, vel_intervals] = values

    return data


def main():
    data = load_dataset()
    cnn = connect_to_db()

    # remove flights without trace points
    data = load_or_process_csv(
        "data_w_complete_trajectory.csv", lambda: process_non_empty_flights(data.copy(), cnn)
    )

    data = load_or_process_csv(
        "data_w_flight_phases_features.csv",
        lambda: process_flight_phases_features(data.copy(), cnn),
    )

    # trajectory features: modo_c, vel_mod, vel_z, delta_modo_c
    data = load_or_process_csv(
        "data_w_trajectory_features.csv", lambda: process_trajectory_features(data.copy(), cnn)
    )

    # climb features: vel_mod, vel_z during 30FL, 60FL, 90FL
    data = load_or_process_csv(
        "data_w_climb_vel.csv", lambda: process_climb_velocity(data.copy(), cnn)
    )

    # final changes
    file_path = os.path.join(PROCESSED_DATA_DIR, "processed_dataset.csv")

    data = data.drop(
        columns=["takeoff_diff", "landing_diff", "haversine_dist", "estimated_cruise_distance"]
    )  # remove useless features

    # if statement to add new features without run all the processing again
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)  # load processed_dataset if it exists before
    else:
        # add mtow from rates table (PERSEO PRO)
        mtows_rates_weight = pd.read_csv(
            os.path.join(RAW_DATA_DIR, "mtow_rates_flows_2022.csv"), sep=";"
        )
        mtows_rates_weight = mtows_rates_weight.drop_duplicates()
        data = data.merge(mtows_rates_weight, on="flightKey", how="left")
        act = data[data["mtow_rates"].isna()].aircraftType.unique()
        for ac in list(act):
            data = fill_mtow_rates(data, ac)
        data["mtow_rates"] = data["mtow_rates"].fillna(
            data["mtow_openap"]
        )  # unmatched mtows fill with openap mtow

        # add seats and payload (PERSEO PRO)
        aircraft_seats = pd.read_csv(os.path.join(RAW_DATA_DIR, "aircraft_seats.csv"), sep=";")
        iata = pd.read_csv(os.path.join(RAW_DATA_DIR, "iata.csv"), sep=";")

        # add iata regions
        unique_combinations_in_data = data[["adep", "ades"]].drop_duplicates()
        iata_filtered = pd.merge(
            iata, unique_combinations_in_data, on=["adep", "ades"], how="inner"
        )
        data = pd.merge(data, iata_filtered, on=["adep", "ades"], how="inner")
        data = pd.merge(
            data,
            aircraft_seats,
            on=["aircraftRegistration", "IATAregionOrigen", "IATAregionDestino"],
            how="left",
        )

        # filter samples
        data["date"] = data["ATOT_track"].str[:10]
        data["dateFrom"] = data["dateFrom"].fillna(data["date"])

        df_past_dates = data[data["dateFrom"] <= data["date"]]
        idx = df_past_dates.groupby("flightKey")["date"].idxmax()
        data = df_past_dates.loc[idx]

        # input missing values
        load_factor_rules = {
            "Africa": {"inter": 72.9, "intra": 73.3},
            "Asia Pacific": {"inter": 84.9, "intra": 84.9},
            "Europe": {"inter": 85.0, "intra": 85.3},
            "South America": {"inter": 84.4, "intra": 84.5},
            "Middle East": {"inter": 81.0, "intra": 81.2},
            "North America": {"inter": 81.0, "intra": 81.0},
        }

        nan_loadfactor_mask = data["LoadFactor"].isna()

        for region_name, factors in load_factor_rules.items():
            # Rule 1: IATAregionOrigen = 'region_name' AND IATAregionDestino != 'region_name'
            inter_regional_condition = (
                (data["IATAregionOrigen"] == region_name)
                & (data["IATAregionDestino"] != region_name)
                & nan_loadfactor_mask
            )  # Only apply to NaNs
            data.loc[inter_regional_condition, "LoadFactor"] = factors["inter"]

            # Rule 2: IATAregionOrigen = 'region_name' AND IATAregionDestino = 'region_name'
            intra_regional_condition = (
                (data["IATAregionOrigen"] == region_name)
                & (data["IATAregionDestino"] == region_name)
                & nan_loadfactor_mask
            )  # Only apply to NaNs
            data.loc[intra_regional_condition, "LoadFactor"] = factors["intra"]

        act_to_fill = data[(data["seats"].isna()) & (data["Payload"].isna())].aircraftType.unique()

        for ac in act_to_fill:
            median_seats_for_ac = data[data["aircraftType"] == ac]["seats"].median()
            data.loc[(data["aircraftType"] == ac) & (data["seats"].isna()), "seats"] = data.loc[
                (data["aircraftType"] == ac) & (data["seats"].isna()), "seats"
            ].fillna(median_seats_for_ac)

        for ac in act_to_fill:
            mask_payload_nan = (data["aircraftType"] == ac) & (data["Payload"].isna())
            seats_for_calculation = data.loc[mask_payload_nan, "seats"]
            loadfactor_for_calculation = data.loc[mask_payload_nan, "LoadFactor"]
            calculated_payload_values = seats_for_calculation * loadfactor_for_calculation
            data.loc[mask_payload_nan, "Payload"] = data.loc[mask_payload_nan, "Payload"].fillna(
                calculated_payload_values
            )

        data = data.drop(
            columns=["IATAregionOrigen", "IATAregionDestino", "LoadFactor", "dateFrom", "dateTo"]
        )
        data.to_csv(file_path, index=False)  # save current state and continue the execution

    fk_to_remove = [18353362, 17977397, 18657899, 17979421, 17660000, 20088120, 19334453, 18890765]

    data = data[
        ~data["flightKey"].isin(fk_to_remove)
    ]  # remove outliers or flights with atypical tow values

    # split the dataset into the 2 models
    processed_data_m1 = process_data_model_1(data)
    processed_data_m2 = process_data_model_2(data)

    processed_data_H = process_data_H(data)
    processed_data_H = processed_data_H.drop(columns=["wake"])  # drop column with one unique value

    processed_data_M = process_data_M(data)
    processed_data_M = processed_data_M.drop(columns=["wake"])  # drop column with one unique value

    processed_data_m1_M = processed_data_m1[processed_data_m1["wake"] == "M"]
    processed_data_m1_M = processed_data_m1_M.drop(
        columns=["wake"]
    )  # drop column with one unique value

    processed_data_m1_H = processed_data_m1[processed_data_m1["wake"] == "H"]
    processed_data_m1_H = processed_data_m1_H.drop(
        columns=["wake"]
    )  # drop column with one unique value

    processed_data_m2_M = processed_data_m2[processed_data_m2["wake"] == "M"]
    processed_data_m2_M = processed_data_m2_M.drop(
        columns=["wake"]
    )  # drop column with one unique value

    processed_data_m2_H = processed_data_m2[processed_data_m2["wake"] == "H"]
    processed_data_m2_H = processed_data_m2_H.drop(
        columns=["wake"]
    )  # drop column with one unique value

    file_path_m1 = os.path.join(PROCESSED_DATA_DIR, "processed_data_m1.csv")
    file_path_m2 = os.path.join(PROCESSED_DATA_DIR, "processed_data_m2.csv")
    file_path_H = os.path.join(PROCESSED_DATA_DIR, "processed_data_H.csv")
    file_path_M = os.path.join(PROCESSED_DATA_DIR, "processed_data_M.csv")
    file_path_m1_M = os.path.join(PROCESSED_DATA_DIR, "processed_data_m1_M.csv")
    file_path_m1_H = os.path.join(PROCESSED_DATA_DIR, "processed_data_m1_H.csv")
    file_path_m2_M = os.path.join(PROCESSED_DATA_DIR, "processed_data_m2_M.csv")
    file_path_m2_H = os.path.join(PROCESSED_DATA_DIR, "processed_data_m2_H.csv")

    processed_data_m1.to_csv(file_path_m1, index=False)
    processed_data_m2.to_csv(file_path_m2, index=False)
    processed_data_H.to_csv(file_path_H, index=False)
    processed_data_M.to_csv(file_path_M, index=False)
    processed_data_m1_M.to_csv(file_path_m1_M, index=False)
    processed_data_m1_H.to_csv(file_path_m1_H, index=False)
    processed_data_m2_M.to_csv(file_path_m2_M, index=False)
    processed_data_m2_H.to_csv(file_path_m2_H, index=False)


if __name__ == "__main__":
    main()
