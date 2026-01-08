######## libraries
import os
import pickle
import mlflow
import optuna
import h2o
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from h2o.automl import H2OAutoML
from atow_estimation.paths import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
from atow_estimation.config import (
    TEST_SIZE,
    N_TRIALS,
    H2O_MAX_RUNTIME_SECS,
    H2O_N_MODELS,
    H2O_ALGOS,
    M1_M_FEATURES,
    M2_M_FEATURES,
    H_FEATURES,
    M_FEATURES,
    CATEGORICAL_VARIABLES_M1_M,
    CATEGORICAL_VARIABLES_M2_M,
    CATEGORICAL_VARIABLES_H,
    CATEGORICAL_VARIABLES_M,
    LIB_CATBOOST,
    LIB_LIGHTGBM,
    LIB_XGBOOST,
    LIB_H2O,
    FIGURE_FEATURE_IMPORTANCE,
    FIGURE_RESIDUALS_DIST,
    FIGURE_SCATTER_RESIDUALS,
    FIGURE_SCATTER,
    FIGURE_PARETO,
    CSV_LEADERBOARD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_USERNAME,
    get_lightgbm_params,
    get_catboost_params,
    get_xgboost_params,
)
from atow_estimation.utils import (
    preprocess_lightgbm,
    preprocess_xgboost,
    preprocess_catboost,
    preprocess_h2o,
    get_model_metrics,
    get_model_figs,
    generate_run_name,
    log_model_mlflow,
)
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor


def load_datasets():
    """Load all required datasets and create MLflow DataFrames."""
    dataset_names = {
        "m1": "Flights with climb phase trajectory",
        "m2": "Flights without climb phase trajectory",
        "M": "All flights (M wake category)",
        "H": "All flights (H wake category)",
    }
    wake_categories = {"M": "M wake category", "H": "H wake category"}
    datasets = {}
    for key, _ in dataset_names.items():
        if key in ["m1", "m2"]:
            for wc_key, _ in wake_categories.items():
                data_name = f"data_{key}_{wc_key}"
                file_name = f"processed_data_{key}_{wc_key}.csv"
                df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, file_name))
                datasets[data_name] = df
        else:  # For 'M' and 'H'
            data_name = f"data_{key}"
            file_name = f"processed_data_{key}.csv"
            df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, file_name))
            datasets[data_name] = df
    return datasets


def open_model(model_name):
    """Opens and deserializes a pickled model."""
    try:
        with open(os.path.join(MODELS_DIR, f"{model_name}.pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file '{model_name}.pkl' not found.")
        return None
    except Exception as e:
        print(f"Error loading model '{model_name}.pkl': {e}")
        return None


def save_model(model, model_name):
    """Serializes and saves a model to a pickle file."""
    try:
        with open(os.path.join(MODELS_DIR, f"{model_name}.pkl"), "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        print(f"Error saving model '{model_name}.pkl': {e}")


def save_figs(
    run_name,
    y_test,
    y_pred,
    lib,
    model_type,
    wake_category,
):
    """Generates and saves model-related figures (residuals distribution, scatter plots)."""
    residuals = y_test - y_pred
    lib_display_names = {
        LIB_LIGHTGBM: "LightGBM",
        LIB_XGBOOST: "XGBoost",
        LIB_CATBOOST: "CatBoost",
    }
    lib_name = lib_display_names.get(lib, "Unknown Library")
    heads = [
        f"{lib_name}: Distribution of Residuals",
        f"{lib_name}: Regression Overview",
        f"{lib_name}: Residuals vs Predicted Values",
    ]
    tail_map = {
        "m1": {
            "M_wake": " of Model 1 (for Medium wake category)",
            "H_wake": " of Model 1 (for Heavy wake category)",
        },
        "m2": {
            "M_wake": " of Model 2 (for Medium wake category)",
            "H_wake": " of Model 2 (for Heavy wake category)",
        },
        "m1+m2": {"M_wake": " (for Medium wake category)", "H_wake": " (for Heavy wake category)"},
    }
    specific_folder = f"{model_type}/{wake_category}"
    if model_type == "m1+m2":
        specific_folder = "M" if wake_category == "M_wake" else "H"

    tail = tail_map.get(model_type, {}).get(wake_category, "")

    get_model_figs(
        residuals,
        y_pred,
        y_test,
        heads[0] + tail,
        heads[1] + tail,
        heads[2] + tail,
        os.path.join(FIGURES_DIR, specific_folder, f"{run_name}{FIGURE_RESIDUALS_DIST}"),
        os.path.join(FIGURES_DIR, specific_folder, f"{run_name}{FIGURE_SCATTER}"),
        os.path.join(FIGURES_DIR, specific_folder, f"{run_name}{FIGURE_SCATTER_RESIDUALS}"),
    )


def print_study_results(study):
    """Prints the best trial information from an Optuna study."""
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best RMSE: {study.best_trial.value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


def print_model_metrics(r2, mae, rmse, mape):
    """Prints the evaluation metrics of a trained model."""
    print(f"  R^2 Score: {r2:.4f}")
    print(f"  Mean Absolute Error: {mae:.2f}")
    print(f"  Root Mean Squared Error: {rmse:.2f}")
    print(f"  Mean Absolute Percentage Error: {mape:.4f}")


def run_hyperparameter_tuning(objective_fn, n_trials, *args):
    """Runs hyperparameter tuning using Optuna."""
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective_fn(trial, *args),
        n_trials=n_trials,
        timeout=21600,  # 6 hours
    )
    print_study_results(study)
    return study


def train_and_evaluate_model(
    run_name,
    model_type,
    wake_category,
    study,
    X_train,
    y_train,
    X_test,
    y_test,
    model_lib,
    model_class,
    categorical_set,
):
    """Trains a model with best hyperparameters and evaluates it."""
    best_params = study.best_trial.params.copy()
    final_model = None

    if model_lib == LIB_LIGHTGBM:
        lgbm_params = {
            **best_params,
            "verbose": -1,
            "objective": "regression_l1",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_jobs": -1,
            "random_state": 42,
        }
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_set)
        test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_set)
        final_model = lgb.train(
            lgbm_params,
            train_data,
            valid_sets=[test_data],
            valid_names=["valid"],
            num_boost_round=10000,
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=True),
                log_evaluation(period=1000),
            ],
        )
    elif model_lib == LIB_CATBOOST:
        cat_params = {
            **best_params,
            "iterations": 10000,
            "random_seed": 42,
            "loss_function": "RMSE",
            "task_type": "CPU",
            "allow_writing_files": False,
            "use_best_model": True,
            "thread_count": -1,
            "od_type": "Iter",
            "od_wait": 50,
        }

        final_model = model_class(**cat_params)

        train_pool = Pool(X_train, y_train, cat_features=categorical_set)
        test_pool = Pool(X_test, y_test, cat_features=categorical_set)

        final_model.fit(
            train_pool,
            eval_set=test_pool,
            verbose=1000,
        )

        feature_importances = final_model.get_feature_importance(train_pool)
        feature_im_df = pd.DataFrame(
            {"feature": X_train.columns, "importance": feature_importances}
        )

        feature_im_df = feature_im_df.sort_values(by="importance", ascending=False)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=feature_im_df[:20], x="importance", y="feature")

        for p in ax.patches:
            width = p.get_width()
            ax.text(
                width + 0.005,  # x-coordinate (slightly offset)
                p.get_y() + p.get_height() / 2,  # y-coordinate (center of bar)
                f"{width:.3f}",  # label text
                va="center",
            )
        max_width = feature_im_df["importance"].head(20).max()
        ax.set_xlim(0, max_width + 1.2)  # Agrega margen derecho

        plt.title("CatBoost Feature importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        specific_folder = f"{model_type}/{wake_category}"
        if model_type == "m1+m2":
            specific_folder = "M" if wake_category == "M_wake" else "H"
        plt.savefig(
            os.path.join(FIGURES_DIR, specific_folder, f"{run_name}{FIGURE_FEATURE_IMPORTANCE}")
        )
        plt.close()

    elif model_lib == LIB_XGBOOST:

        # XGBoost parameters
        xgb_params = {
            **best_params,
            "n_estimators": 10000,
            "random_state": 42,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "eval_metric": "rmse",
            "tree_method": "hist",
        }

        final_model = model_class(**xgb_params, early_stopping_rounds=50)

        final_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=1000,
        )
    else:
        raise ValueError(f"Unsupported model library: {model_lib}")
    y_pred = final_model.predict(X_test)
    r2, mae, rmse, mape = get_model_metrics(y_pred, y_test)
    print_model_metrics(r2, mae, rmse, mape)
    save_model(final_model, run_name)
    save_figs(run_name, y_test, y_pred, model_lib, model_type, wake_category)
    return final_model


def train_and_evaluate_h2o_model(
    train, test, features, target, run_name, model_type, wake_category
):
    aml = H2OAutoML(
        max_runtime_secs=H2O_MAX_RUNTIME_SECS,
        max_models=H2O_N_MODELS,
        seed=42,
        sort_metric="RMSE",
        include_algos=H2O_ALGOS,
        verbosity="info",
    )
    aml.train(x=features, y=target, training_frame=train, leaderboard_frame=test)

    leaderboard = aml.leaderboard
    best_model = aml.leader
    predictions = best_model.predict(test).as_data_frame().values.flatten()
    y_test = test[target].as_data_frame().values.flatten()

    r2, mae, rmse, mape = get_model_metrics(predictions, y_test)
    print_model_metrics(r2, mae, rmse, mape)
    # save_model(best_model, run_name)
    h2o.save_model(model=best_model, path=MODELS_DIR, force=True, filename=run_name)
    print(f"\nGenerating and saving the Pareto Front plot to '{run_name}{FIGURE_PARETO}'...")

    specific_folder = f"{model_type}/{wake_category}"
    if model_type == "m1+m2":
        specific_folder = "M" if wake_category == "M_wake" else "H"

    combined_leaderboard = h2o.make_leaderboard([aml], test, extra_columns="ALL")
    pf = h2o.explanation.pareto_front(
        combined_leaderboard,
        x_metric="predict_time_per_row_ms",
        y_metric="rmse",
        optimum="bottom left",
    )
    pareto_figure = pf.figure()
    plt.savefig(os.path.join(FIGURES_DIR, specific_folder, f"{run_name}{FIGURE_PARETO}"))
    print("Plot saved successfully.")
    print(f"\nGenerating and saving the leaderboard to '{run_name}{CSV_LEADERBOARD}'...")
    leaderboard_df = leaderboard.as_data_frame()
    leaderboard_df.to_csv(
        os.path.join(FIGURES_DIR, specific_folder, f"{run_name}{CSV_LEADERBOARD}"), index=False
    )
    save_figs(run_name, y_test, predictions, "h2o", model_type, wake_category)
    return best_model


def run_model_pipeline(
    data_key,
    preprocess_fn,
    model_lib,
    model_class,
    objective_fn,
    model_type,
    wake_category,
    categorical_set,
    features,
    name_prefix,
    wc_name,
    data_all,
):
    """Runs the full pipeline for a single model: preprocessing, hyperparameter tuning, training, saving, and MLflow logging."""
    run_name = generate_run_name(model_lib)
    data = data_all[data_key]

    if model_lib == "h2o":
        train_h2o, test_h2o, X_test_df, y_test_np = preprocess_fn(data[features])
        target = "tow"
        model = train_and_evaluate_h2o_model(
            train=train_h2o,
            test=test_h2o,
            features=[col for col in train_h2o.columns if col != target],
            target=target,
            run_name=run_name,
            model_type=model_type,
            wake_category=wake_category,
        )

        # Log model with MLflow
        dataset = mlflow.data.from_pandas(
            data[features], name=f"{name_prefix} ({wc_name})", targets="tow"
        )
        print("Logging H2O model to MLflow...")

        log_model_mlflow(
            data[features].drop(["tow"], axis=1),
            data[features]["tow"],
            dataset,
            model,
            model_lib,
            model_type,
            wake_category,
            run_name,
            test_h2o,
            None,
            TEST_SIZE,
        )
        print("MLflow logging complete.")
        return  # Exit after H2O path

    X_train, X_test, y_train, y_test, *pools = preprocess_fn(data[features])
    tuning_args = None

    if model_lib == LIB_CATBOOST:
        train_pool = pools[0]
        test_pool = pools[1]
        tuning_args = (train_pool, test_pool)
    elif model_lib == LIB_LIGHTGBM:
        tuning_args = (X_train, X_test, y_train, y_test, categorical_set)
    else:
        tuning_args = (X_train, X_test, y_train, y_test)

    # Hyperparameter tuning
    print(f"Starting hyperparameter tuning for {model_lib} {model_type} {wake_category}...")
    study = run_hyperparameter_tuning(objective_fn, N_TRIALS, *tuning_args)
    print("Hyperparameter tuning finished.")

    # Training and evaluation
    print(f"Starting final model training for {model_lib} {model_type} {wake_category}...")
    final_model = train_and_evaluate_model(
        run_name=run_name,
        model_type=model_type,
        wake_category=wake_category,
        study=study,
        X_train=X_train,  # Pass X_train
        y_train=y_train,  # Pass y_train
        X_test=X_test,
        y_test=y_test,
        model_lib=model_lib,
        model_class=model_class,
        categorical_set=categorical_set,
    )
    print("Final model training and evaluation finished.")

    # MLflow logging
    dataset = mlflow.data.from_pandas(
        data[features], name=f"{name_prefix} ({wc_name})", targets="tow"
    )
    print("Logging model to MLflow...")
    log_model_mlflow(
        data[features].drop(["tow"], axis=1),
        data[features]["tow"],
        dataset,
        final_model,
        model_lib,
        model_type,
        wake_category,
        run_name,
        X_test,
        y_test,
        TEST_SIZE,
    )
    print("MLflow logging complete.")


def objective_lightgbm(trial, X_train, X_test, y_train, y_test, categorical_set):
    params = get_lightgbm_params(trial)
    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_set, free_raw_data=False
    )
    test_data = lgb.Dataset(
        X_test, label=y_test, categorical_feature=categorical_set, free_raw_data=False
    )

    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        valid_names=["valid"],
        num_boost_round=10_000,
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=1000)],
    )
    return model.best_score["valid"]["rmse"]


def objective_catboost(trial, train_pool, test_pool):
    params = get_catboost_params(trial)
    model = CatBoostRegressor(**params)
    model.fit(
        train_pool, eval_set=test_pool, early_stopping_rounds=params["od_wait"], verbose=1000
    )
    best_rmse_on_val = model.get_best_score()["validation"]["RMSE"]
    trial.set_user_attr("best_iteration", model.get_best_iteration())
    return best_rmse_on_val


def objective_xgboost(trial, X_train, X_test, y_train, y_test):
    params = get_xgboost_params(trial)
    model = XGBRegressor(**params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=1000)
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    trial.set_user_attr("best_iteration", model.best_iteration)
    return rmse


def main():
    """Main function to set up environment, load data, and run model training pipelines."""
    # Setting environment variables for MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT_NAME
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Loading datasets...")
    data = load_datasets()
    print("Datasets loaded.")

    # Define model configurations to iterate over
    model_configs = [
        # LightGBM Models
        # {
        #     "data_key": "data_m1_M",
        #     "preprocess_fn": preprocess_lightgbm,
        #     "model_lib": LIB_LIGHTGBM,
        #     "model_class": lgb.LGBMRegressor,
        #     "objective_fn": objective_lightgbm,
        #     "model_type": "m1",
        #     "wake_category": "M_wake",
        #     "categorical_set": CATEGORICAL_VARIABLES_M1_M,
        #     "features": M1_M_FEATURES,
        #     "name_prefix": "Flights with climb phase trajectory",
        #     "wc_name": "M wake category",
        # },
        # {
        #     "data_key": "data_m2_M",
        #     "preprocess_fn": preprocess_lightgbm,
        #     "model_lib": LIB_LIGHTGBM,
        #     "model_class": lgb.LGBMRegressor,
        #     "objective_fn": objective_lightgbm,
        #     "model_type": "m2",
        #     "wake_category": "M_wake",
        #     "categorical_set": CATEGORICAL_VARIABLES_M2_M,
        #     "features": M2_M_FEATURES,
        #     "name_prefix": "Flights without climb phase trajectory",
        #     "wc_name": "M wake category",
        # },
        # {
        #     "data_key": "data_M",
        #     "preprocess_fn": preprocess_lightgbm,
        #     "model_lib": LIB_LIGHTGBM,
        #     "model_class": lgb.LGBMRegressor,
        #     "objective_fn": objective_lightgbm,
        #     "model_type": "m1+m2",
        #     "wake_category": "M_wake",
        #     "categorical_set": CATEGORICAL_VARIABLES_M,
        #     "features": M_FEATURES,
        #     "name_prefix": "All flights",
        #     "wc_name": "M wake category",
        # },
        # {
        #     "data_key": "data_H",
        #     "preprocess_fn": preprocess_lightgbm,
        #     "model_lib": LIB_LIGHTGBM,
        #     "model_class": lgb.LGBMRegressor,
        #     "objective_fn": objective_lightgbm,
        #     "model_type": "m1+m2",
        #     "wake_category": "H_wake",
        #     "categorical_set": CATEGORICAL_VARIABLES_H,
        #     "features": H_FEATURES,
        #     "name_prefix": "All flights",
        #     "wc_name": "H wake category",
        # },
        # CatBoost Models
        # {
        #     "data_key": "data_m1_M",
        #     "preprocess_fn": preprocess_catboost,
        #     "model_lib": LIB_CATBOOST,
        #     "model_class": CatBoostRegressor,
        #     "objective_fn": objective_catboost,
        #     "model_type": "m1",
        #     "wake_category": "M_wake",
        #     "categorical_set": CATEGORICAL_VARIABLES_M1_M,
        #     "features": M1_M_FEATURES,
        #     "name_prefix": "Flights with climb phase trajectory",
        #     "wc_name": "M wake category",
        # },
        #     {
        #         "data_key": "data_m2_M",
        #         "preprocess_fn": preprocess_catboost,
        #         "model_lib": LIB_CATBOOST,
        #         "model_class": CatBoostRegressor,
        #         "objective_fn": objective_catboost,
        #         "model_type": "m2",
        #         "wake_category": "M_wake",
        #         "categorical_set": CATEGORICAL_VARIABLES_M2_M,
        #         "features": M2_M_FEATURES,
        #         "name_prefix": "Flights without climb phase trajectory",
        #         "wc_name": "M wake category",
        #     },
        #     {
        #         "data_key": "data_M",
        #         "preprocess_fn": preprocess_catboost,
        #         "model_lib": LIB_CATBOOST,
        #         "model_class": CatBoostRegressor,
        #         "objective_fn": objective_catboost,
        #         "model_type": "m1+m2",
        #         "wake_category": "M_wake",
        #         "categorical_set": CATEGORICAL_VARIABLES_M,
        #         "features": M_FEATURES,
        #         "name_prefix": "All flights",
        #         "wc_name": "M wake category",
        #     },
        #     {
        #         "data_key": "data_H",
        #         "preprocess_fn": preprocess_catboost,
        #         "model_lib": LIB_CATBOOST,
        #         "model_class": CatBoostRegressor,
        #         "objective_fn": objective_catboost,
        #         "model_type": "m1+m2",
        #         "wake_category": "H_wake",
        #         "categorical_set": CATEGORICAL_VARIABLES_H,
        #         "features": H_FEATURES,
        #         "name_prefix": "All flights",
        #         "wc_name": "H wake category",
        #     },
        #     # XGBoost Models
        #     {
        #         "data_key": "data_m1_M",
        #         "preprocess_fn": preprocess_xgboost,
        #         "model_lib": LIB_XGBOOST,
        #         "model_class": XGBRegressor,
        #         "objective_fn": objective_xgboost,
        #         "model_type": "m1",
        #         "wake_category": "M_wake",
        #         "categorical_set": CATEGORICAL_VARIABLES_M1_M,
        #         "features": M1_M_FEATURES,
        #         "name_prefix": "Flights with climb phase trajectory",
        #         "wc_name": "M wake category",
        #     },
        #     {
        #         "data_key": "data_m2_M",
        #         "preprocess_fn": preprocess_xgboost,
        #         "model_lib": LIB_XGBOOST,
        #         "model_class": XGBRegressor,
        #         "objective_fn": objective_xgboost,
        #         "model_type": "m2",
        #         "wake_category": "M_wake",
        #         "categorical_set": CATEGORICAL_VARIABLES_M2_M,
        #         "features": M2_M_FEATURES,
        #         "name_prefix": "Flights without climb phase trajectory",
        #         "wc_name": "M wake category",
        #     },
        #     {
        #         "data_key": "data_M",
        #         "preprocess_fn": preprocess_xgboost,
        #         "model_lib": LIB_XGBOOST,
        #         "model_class": XGBRegressor,
        #         "objective_fn": objective_xgboost,
        #         "model_type": "m1+m2",
        #         "wake_category": "M_wake",
        #         "categorical_set": CATEGORICAL_VARIABLES_M,
        #         "features": M_FEATURES,
        #         "name_prefix": "All flights",
        #         "wc_name": "M wake category",
        #     },
        #     {
        #         "data_key": "data_H",
        #         "preprocess_fn": preprocess_xgboost,
        #         "model_lib": LIB_XGBOOST,
        #         "model_class": XGBRegressor,
        #         "objective_fn": objective_xgboost,
        #         "model_type": "m1+m2",
        #         "wake_category": "H_wake",
        #         "categorical_set": CATEGORICAL_VARIABLES_H,
        #         "features": H_FEATURES,
        #         "name_prefix": "All flights",
        #         "wc_name": "H wake category",
        #     },
    ]

    h2o_model_configs = [
        # h2o AutoML
        {
            "data_key": "data_m1_M",
            "preprocess_fn": preprocess_h2o,
            "model_lib": LIB_H2O,
            "model_class": None,
            "objective_fn": None,
            "model_type": "m1",
            "wake_category": "M_wake",
            "categorical_set": CATEGORICAL_VARIABLES_M1_M,
            "features": M1_M_FEATURES,
            "name_prefix": "Flights with climb phase trajectory",
            "wc_name": "M wake category",
        },
        {
            "data_key": "data_m2_M",
            "preprocess_fn": preprocess_h2o,
            "model_lib": LIB_H2O,
            "model_class": None,
            "objective_fn": None,
            "model_type": "m2",
            "wake_category": "M_wake",
            "categorical_set": CATEGORICAL_VARIABLES_M2_M,
            "features": M2_M_FEATURES,
            "name_prefix": "Flights without climb phase trajectory",
            "wc_name": "M wake category",
        },
        # {
        #     "data_key": "data_M",
        #     "preprocess_fn": preprocess_h2o,
        #     "model_lib": LIB_H2O,
        #     "model_class": None,
        #     "objective_fn": None,
        #     "model_type": "m1+m2",
        #     "wake_category": "M_wake",
        #     "categorical_set": CATEGORICAL_VARIABLES_M,
        #     "features": M_FEATURES,
        #     "name_prefix": "All flights",
        #     "wc_name": "M wake category",
        # },
        {
            "data_key": "data_H",
            "preprocess_fn": preprocess_h2o,
            "model_lib": LIB_H2O,
            "model_class": None,
            "objective_fn": None,
            "model_type": "m1+m2",
            "wake_category": "H_wake",
            "categorical_set": CATEGORICAL_VARIABLES_H,
            "features": H_FEATURES,
            "name_prefix": "All flights",
            "wc_name": "H wake category",
        },
    ]

    for config in model_configs:
        print(
            f"\n--- Running pipeline for {config['model_lib']} - {config['model_type']} - {config['wake_category']} ---"
        )
        run_model_pipeline(**config, data_all=data)
        print(
            f"--- Finished pipeline for {config['model_lib']} - {config['model_type']} - {config['wake_category']} ---\n"
        )

    h2o.init(nthreads=-1)

    for config in h2o_model_configs:
        print(
            f"\n--- Running pipeline for {config['model_lib']} - {config['model_type']} - {config['wake_category']} ---"
        )
        run_model_pipeline(**config, data_all=data)
        print(
            f"--- Finished pipeline for {config['model_lib']} - {config['model_type']} - {config['wake_category']} ---\n"
        )


if __name__ == "__main__":
    main()
