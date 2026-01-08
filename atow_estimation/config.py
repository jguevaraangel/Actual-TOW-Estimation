# Store useful variables and configuration
import mlflow
from catboost import metrics

# --- Training params ---
TEST_SIZE = 0.2
N_TRIALS = 20
H2O_MAX_RUNTIME_SECS = None
H2O_N_MODELS = 10
H2O_ALGOS = ["GBM", "XGBoost", "DRF", "DeepLearning", "StackedEnsemble"]
CATEGORICAL_VARIABLES_M1_M = [
    "RECATwake",
    "aircraftType",
    "airlineCode",
]

CATEGORICAL_VARIABLES_M2_M = [
    "routeType",
    "RECATwake",
    "aircraftType",
    "airlineCode",
]

CATEGORICAL_VARIABLES_H = [
    "routeType",
    "RECATwake",
    "aircraftType",
    "airlineCode",
]

CATEGORICAL_VARIABLES_M = [
    "routeType",
    "flightType",
    "RECATwake",
    "aircraftType",
    "airlineCode",
]


def get_catboost_params(trial):
    params = {
        # model params
        "iterations": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "random_strength": trial.suggest_float("random_strength", 10, 50, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 15),
        # metrics params
        "eval_metric": metrics.RMSE(),
        "random_seed": 42,
        "task_type": "CPU",
        "thread_count": -1,
        "use_best_model": True,
        "od_type": "Iter",
        "od_wait": 50,
    }
    return params


def get_lightgbm_params(trial):
    params = {
        # model params
        "n_estimators": 10000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "min_child_weight": trial.suggest_int("min_child_weight", 1e-5, 10),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        # metrics params
        "objective": "regression_l1",
        "metric": "rmse",
        "n_jobs": -1,
        "device": "cpu",
        "boosting_type": "gbdt",
        "random_state": 42,
        "verbosity": -1,
    }
    return params


def get_xgboost_params(trial):
    params = {
        # model params
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        # metrics params
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "rmse",
        "tree_method": "hist",
    }
    return params


# --- Tracking mlflow server ---
MLFLOW_TRACKING_USERNAME = "jaguevara"
MLFLOW_TRACKING_PASSWORD = "jaguevara_mlflow"
MLFLOW_TRACKING_URI = "http://192.168.25.110:5000"
MLFLOW_EXPERIMENT_NAME = "HEAVY_atow_estimation"

# --- Models ---
LIB_LIGHTGBM = "lightgbm"
LIB_XGBOOST = "xgboost"
LIB_CATBOOST = "catboost"
LIB_H2O = "h2o"

# --- Figures ---
FIGURE_RESIDUALS_DIST = "_residuals_distribution.png"
FIGURE_SCATTER_RESIDUALS = "_scatter_residuals.png"
FIGURE_SCATTER = "_scatter.png"
FIGURE_FEATURE_IMPORTANCE = "_feature_importance.png"
FIGURE_PARETO = "_pareto.png"
CSV_LEADERBOARD = "_leaderboard.csv"
ARTIFACT_SUBFOLDER_FIGURES = "Figures"  # Subfolder in MLflow for figures
ARTIFACT_SUBFOLDER_LEADERBOARD = "Leaderboard"

# --- Model Logger Mapping ---
MODEL_LOGGERS = {
    LIB_LIGHTGBM: mlflow.lightgbm.log_model,
    LIB_XGBOOST: mlflow.xgboost.log_model,
    LIB_CATBOOST: mlflow.catboost.log_model,
    LIB_H2O: mlflow.h2o.log_model,
}

# --- Feature Selection ---
M1_M_FEATURES = [
    "ADEPLat",
    "ADEPLong",
    "ADESLat",
    "ADESLong",
    "aircraftType",
    "airlineCode",
    "RECATwake",
    "cruiseSpeed",
    "cruiseLevel",
    "ATOT_track_hour",
    "ATOT_track_day",
    "ALDT_track_day",
    "flight_duration",
    "CLIMB_duration",
    "CRUISE_duration",
    "total_distance",
    "CLIMB_distance",
    "CRUISE_distance",
    "CLIMB_modo_c_variance",
    "CLIMB_vel_mod_variance",
    "CLIMB_vel_z_median",
    "CLIMB_vel_z_variance",
    "CLIMB_delta_modo_c",
    "CRUISE_modo_c_median",
    "CLIMB_vel_mod_mean_0_30fl",
    "CLIMB_vel_z_mean_0_30fl",
    "CLIMB_vel_z_variance_0_30fl",
    "CLIMB_vel_mod_mean_0_60fl",
    "CLIMB_vel_z_mean_0_60fl",
    "CLIMB_vel_mod_variance_0_60fl",
    "CLIMB_vel_mod_mean_0_90fl",
    "CLIMB_vel_z_mean_0_90fl",
    "CLIMB_vel_mod_variance_0_90fl",
    "CLIMB_vel_mod_mean_30_60fl",
    "CLIMB_vel_mod_mean_30_90fl",
    "CLIMB_vel_z_mean_30_90fl",
    "CLIMB_vel_z_mean_60_90fl",
    "CLIMB_vel_z_variance_60_90fl",
    "mtow_openap",
    "mlw",
    "mfc",
    "oew",
    "mtow_rates",
    "seats",
    "Payload",
    "tow",
]

M2_M_FEATURES = [
    "ADEPLat",
    "ADEPLong",
    "ADESLat",
    "ADESLong",
    "routeType",
    "aircraftType",
    "airlineCode",
    "RECATwake",
    "cruiseSpeed",
    "cruiseLevel",
    "ATOT_track_hour",
    "ATOT_track_day",
    "ALDT_track_hour",
    "ALDT_track_day",
    "flight_duration",
    "CRUISE_duration",
    "total_distance",
    "CRUISE_distance",
    "DESCENT_distance",
    "CRUISE_modo_c_median",
    "CRUISE_vel_mod_median",
    "CRUISE_vel_mod_variance",
    "DESCENT_modo_c_variance",
    "DESCENT_vel_mod_median",
    "DESCENT_vel_mod_variance",
    "DESCENT_vel_z_median",
    "DESCENT_vel_z_variance",
    "DESCENT_delta_modo_c",
    "mtow_openap",
    "mlw",
    "mfc",
    "oew",
    "mtow_rates",
    "seats",
    "Payload",
    "tow",
]

H_FEATURES = [
    "ADEPLat",
    "ADEPLong",
    "ADESLat",
    "ADESLong",
    "routeType",
    "aircraftType",
    "airlineCode",
    "RECATwake",
    "cruiseSpeed",
    "cruiseLevel",
    "ATOT_track_hour",
    "ATOT_track_day",
    "ALDT_track_hour",
    "ALDT_track_day",
    "flight_duration",
    "CRUISE_duration",
    "total_distance",
    "CRUISE_distance",
    "CLIMB_modo_c_variance",
    "CLIMB_vel_z_median",
    "CRUISE_modo_c_median",
    "CRUISE_vel_mod_median",
    "mtow_openap",
    "mlw",
    "mfc",
    "oew",
    "mtow_rates",
    "seats",
    "Payload",
    "tow",
]

M_FEATURES = [
    "ADEPLat",
    "ADEPLong",
    "ADESLat",
    "ADESLong",
    "routeType",
    "aircraftType",
    "airlineCode",
    "flightType",
    "RECATwake",
    "cruiseSpeed",
    "cruiseLevel",
    "ATOT_track_hour",
    "ATOT_track_day",
    "ALDT_track_hour",
    "ALDT_track_day",
    "flight_duration",
    "CLIMB_duration",
    "CRUISE_duration",
    "total_distance",
    "CLIMB_distance",
    "CRUISE_distance",
    "CLIMB_modo_c_variance",
    "CLIMB_vel_mod_variance",
    "CLIMB_vel_z_median",
    "CLIMB_vel_z_variance",
    "CLIMB_delta_modo_c",
    "CRUISE_modo_c_median",
    "CRUISE_vel_mod_median",
    "CRUISE_vel_mod_variance",
    "CRUISE_delta_modo_c",
    "DESCENT_modo_c_variance",
    "DESCENT_vel_mod_variance",
    "DESCENT_vel_z_variance",
    "DESCENT_delta_modo_c",
    "mtow_openap",
    "mlw",
    "mfc",
    "oew",
    "mtow_rates",
    "seats",
    "Payload",
    "tow",
]
