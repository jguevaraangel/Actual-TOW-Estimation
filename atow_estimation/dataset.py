import os
import pandas as pd
from openap import prop
from atow_estimation.paths import RAW_DATA_DIR, INTERIM_DATA_DIR, EXTERNAL_DATA_DIR, REPORTS_DIR


def load_datasets():
    """Load all required datasets with appropriate datatypes."""
    datasets = {
        "flight_info": pd.read_csv(
            os.path.join(RAW_DATA_DIR, "flowflight_arrival_departure_airline_202201_202301.csv"),
            sep=";",
            dtype={31: str},
        ),
        "flight_phases": pd.read_csv(
            os.path.join(RAW_DATA_DIR, "flowflightphases_202201_202301 1.csv"), sep=";"
        ),
        "challenge_set": pd.read_csv(os.path.join(RAW_DATA_DIR, "challenge_set.csv")),
        "airline_id": pd.read_csv(
            os.path.join(RAW_DATA_DIR, "Airlines_id.txt"),
            sep="\t",
            header=None,
            names=["hash", "some_value", "some_bool", "airline_name", "airline"],
        ),
        "cfmu_dim": pd.read_csv(
            os.path.join(RAW_DATA_DIR, "cfmuflight_arrival_departure_202201_202301 1.csv"),
            sep=";",
            dtype={14: str, 15: str, 16: str, 17: str, 18: str, 26: str, 27: str, 28: str},
        ),
    }
    return datasets


def process_flight_info(flight_info):
    """Process flight information data."""
    # Filter commercial flights and rename column
    flight_info = flight_info.query("flightType != 'GENERAL' and flightType != 'MILITARY'").rename(
        columns={"aircraft": "aircraftType"}
    )

    # Format date columns
    for col in ["ATOT_track", "ALDT_track"]:
        flight_info[col] = flight_info[col].str.slice(0, 19)

    return flight_info


def process_challenge_set(challenge_set, airline_id):
    """Process the challenge set data and merge with airline information."""
    return (
        challenge_set.merge(
            airline_id[["hash", "airline"]], left_on="airline", right_on="hash", how="left"
        )
        .drop(columns=["hash"])
        .rename(
            columns={
                "airline_x": "airline_hash",
                "airline_y": "airlineCode",
                "actual_offblock_time": "ATOT_track",
                "arrival_time": "ALDT_track",
                "aircraft_type": "aircraftType",
            }
        )
    )


def process_cfmu_data(cfmu_dim):
    """Process CFMU flight data with complex column handling."""
    # Process first row separately
    cfmu_first_row = cfmu_dim.iloc[:1].drop(
        columns=["CFMUflightKey", "callsign", "adep", "ades", "ADEPLat", "ADEPLong"]
    )
    cfmu_first_row = cfmu_first_row.rename(
        columns={
            col + "_1": col
            for col in ["CFMUflightKey", "callsign", "adep", "ades", "ADEPLat", "ADEPLong"]
        }
    )

    # Process remaining rows
    cfmu_rest_row = cfmu_dim.iloc[1:].drop(
        columns=[
            "LOBTtimeKey",
            "ALDTdateKey",
            "ALDTtimeKey",
            "ATOT_instant",
            "LOBT_instant",
            "ALDT_instant",
        ]
    )

    cfmu_rest_row = cfmu_rest_row.rename(
        columns={
            "CFMUflightKey_1": "ADESLat",
            "callsign_1": "ADESLong",
            "adep_1": "flightRuleType",
            "ades_1": "routeType",
            "ADEPLat_1": "aircraftType",
            "ADEPLong_1": "aircraftOperator",
            "ADESLat": "operatingAircraftOperator",
            "ADESLong": "ATOTdateKey",
            "flightRuleType": "ATOTTimeKey",
            "routeType": "LOBTDateKey",
            "aircraftType": "LOBTTimeKey",
            "aircraftOperator": "ALDTDateKey",
            "operatingAircraftOperator": "ALDTTimeKey",
            "ATOTdateKey": "ATOT_instant",
            "ATOTtimeKey": "LOBT_instant",
            "LOBTdateKey": "ALDT_instant",
        }
    )

    # Combine rows and standardize column names
    cfmu_dim = pd.concat([cfmu_first_row, cfmu_rest_row], ignore_index=True).rename(
        columns={
            "flowsFlightKey": "flightKey",
            "ATOTdateKey": "ATOTDateKey",
            "ATOTtimeKey": "ATOTTimeKey",
            "ATOT_instant": "ATOT_track",
            "ALDTdateKey": "ALDTDateKey",
            "ALDTtimeKey": "ALDTTimeKey",
            "ALDT_instant": "ALDT_track",
            "LOBTdateKey": "LOBTDateKey",
            "LOBTtimeKey": "LOBTTimeKey",
            "LOBT_instant": "LOBT_track",
        }
    )

    # Data type conversion and additional processing
    cfmu_dim["flightKey"] = cfmu_dim["flightKey"].astype("Int64")
    cfmu_dim["airlineCode"] = cfmu_dim["callsign"].str.slice(0, 3)
    cfmu_dim = cfmu_dim.drop_duplicates(subset=["flightKey"], keep="first")

    # Format date columns
    for col in ["ATOT_track", "ALDT_track"]:
        cfmu_dim[col] = cfmu_dim[col].str.slice(0, 19)

    return cfmu_dim


def process_flight_phases(flight_phases):
    """Transform flight phases data into a wide format."""
    # Filter and pivot the data
    flight_phases = flight_phases[flight_phases["flightKey"] != 0][
        ["flightKey", "phaseName", "dtPhaseStart", "dtPhaseEnd"]
    ].pivot(index="flightKey", columns="phaseName", values=["dtPhaseStart", "dtPhaseEnd"])

    # Flatten column names and reset index
    flight_phases.columns = [f"{x}_{y}" for x, y in flight_phases.columns]
    return flight_phases.reset_index()


def main():
    # Load all datasets
    datasets = load_datasets()

    # Create folders to store figures
    if not os.path.exists(os.path.join(REPORTS_DIR, "figures", "H")):
        os.makedirs(os.path.join(REPORTS_DIR, "figures", "H"))

    if not os.path.exists(os.path.join(REPORTS_DIR, "figures", "M")):
        os.makedirs(os.path.join(REPORTS_DIR, "figures", "M"))

    if not os.path.exists(os.path.join(REPORTS_DIR, "figures", "m1")):
        os.makedirs(os.path.join(REPORTS_DIR, "figures", "m1"))
        if not os.path.exists(os.path.join(REPORTS_DIR, "figures", "m1", "M_wake")):
            os.makedirs(os.path.join(REPORTS_DIR, "figures", "m1", "M_wake"))
        if not os.path.exists(os.path.join(REPORTS_DIR, "figures", "m1", "H_wake")):
            os.makedirs(os.path.join(REPORTS_DIR, "figures", "m1", "H_wake"))

    if not os.path.exists(os.path.join(REPORTS_DIR, "figures", "m2")):
        os.makedirs(os.path.join(REPORTS_DIR, "figures", "m2"))
        if not os.path.exists(os.path.join(REPORTS_DIR, "figures", "m2", "M_wake")):
            os.makedirs(os.path.join(REPORTS_DIR, "figures", "m2", "M_wake"))
        if not os.path.exists(os.path.join(REPORTS_DIR, "figures", "m2", "H_wake")):
            os.makedirs(os.path.join(REPORTS_DIR, "figures", "m2", "H_wake"))

    # Process flight information
    flight_info = process_flight_info(datasets["flight_info"])

    # Process challenge set
    challenge_set = process_challenge_set(datasets["challenge_set"], datasets["airline_id"])

    # Format challenge set date column
    challenge_set["ATOT_track"] = (
        challenge_set["ATOT_track"].str.slice(0, 19).str.replace("T", " ")
    )

    # Process CFMU data
    cfmu_dim = process_cfmu_data(datasets["cfmu_dim"])

    # Select columns for merging
    final_challenge_set = challenge_set[
        ["airlineCode", "adep", "ades", "aircraftType", "ATOT_track", "tow"]
    ].copy()
    final_flight_info_cfmu = cfmu_dim.copy()

    # Convert to datetime for proper joining
    for df in [final_flight_info_cfmu, final_challenge_set]:
        df["ATOT_track"] = pd.to_datetime(df["ATOT_track"])

    # Sort for asof merge
    final_flight_info_cfmu = final_flight_info_cfmu.sort_values(by=["ATOT_track"])
    final_challenge_set = final_challenge_set.sort_values(by=["ATOT_track"])

    # Perform asof merge to match records within time tolerance
    merged = pd.merge_asof(
        left=final_flight_info_cfmu,
        right=final_challenge_set,
        on="ATOT_track",
        by=["airlineCode", "adep", "ades", "aircraftType"],
        tolerance=pd.Timedelta("60min"),
    )

    # Remove rows with missing target values
    data = merged.dropna(subset=["tow"])

    # Merge with flight information
    data_w_flight_info = data.merge(flight_info, on=["flightKey"], how="left")

    # Remove redundant columns
    cols_to_drop = [
        "callsign_y",
        "adep_y",
        "ades_y",
        "aircraftType_y",
        "ATOTDateKey_y",
        "ATOTTimeKey_y",
        "ATOT_track_y",
        "ALDTDateKey_y",
        "ALDTTimeKey_y",
        "ALDT_track_y",
        "airlineCode_y",
    ]
    data = data_w_flight_info.drop(columns=cols_to_drop).rename(
        columns=lambda x: x.replace("_x", "")
    )

    # Process flight phases
    flight_phases = process_flight_phases(datasets["flight_phases"])

    # Merge phases
    data = data.merge(flight_phases, on=["flightKey"], how="left")

    # Fill missing phase times
    data["dtPhaseStart_CLIMB"] = data["dtPhaseStart_CLIMB"].fillna(data["ATOT_track"])
    data["dtPhaseEnd_DESCENT"] = data["dtPhaseEnd_DESCENT"].fillna(data["ALDT_track"])

    # Ensure correct formatting for phase timestamps
    for col in [
        "dtPhaseStart_CLIMB",
        "dtPhaseEnd_CLIMB",
        "dtPhaseStart_CRUISE",
        "dtPhaseEnd_CRUISE",
        "dtPhaseStart_DESCENT",
        "dtPhaseEnd_DESCENT",
    ]:
        data[col] = data[col].astype(str).str.slice(0, 19)

    # Merge aircraft type masses
    # data = data.merge(datasets['aircraft_type_masses'], on=["aircraftType"], how='left')

    # Convert coordinate values
    for col in ["ADEPLat", "ADEPLong", "ADESLat", "ADESLong"]:
        data[col] = data[col].astype(str).str.replace(",", ".", regex=True).astype(float)

    # Remove unnecessary columns
    cols_to_remove = [
        "ATOTDateKey",
        "ATOTTimeKey",
        "LOBTDateKey",
        "LOBTTimeKey",
        "ALDTDateKey",
        "ALDTTimeKey",
        "LOBT_track",
        "ELDT",
        "ETOT",
        "CTOT",
        "IOBT",
        "flightRule",
        "flightRuleType",
        "processDateReference",
    ]
    data = data.drop(columns=cols_to_remove).dropna(subset=["flightKey", "wake"])

    # OpenAP data
    aircrafts = data["aircraftType"].unique()
    aircrafts = [ac for ac in aircrafts if ac not in ["BCS3", "BCS1"]]
    mtows, mlws, mfcs, oews = [], [], [], []

    for ac in aircrafts:
        ac_openap = prop.aircraft(ac, use_synonym=True)
        mtows.append(ac_openap["mtow"])
        mlws.append(ac_openap["mlw"])
        mfcs.append(ac_openap["mfc"])
        oews.append(ac_openap["oew"])

    # Add BCS3 and BCS1 manually
    aircrafts = aircrafts + ["BCS3", "BCS1"]

    # BCS3
    mtows.append(70900)
    mlws.append(61000)
    mfcs.append(21508)
    oews.append(38241)

    # BCS1
    mtows.append(63700)
    mlws.append(54700)
    mfcs.append(21805)
    oews.append(36809)

    aircrafttype_masses = pd.DataFrame(
        {"aircraftType": aircrafts, "mtow_openap": mtows, "mlw": mlws, "mfc": mfcs, "oew": oews}
    )

    aircrafttype_masses.to_csv(
        os.path.join(EXTERNAL_DATA_DIR, "aircrafttype_masses.csv"), index=False
    )  # save in external data folder
    data = data.merge(aircrafttype_masses, how="inner", on="aircraftType")

    # Save the processed dataset
    data.to_csv(os.path.join(INTERIM_DATA_DIR, "initial_data.csv"), index=False)


if __name__ == "__main__":
    main()
