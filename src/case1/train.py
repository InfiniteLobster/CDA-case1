#--------------------Libraries--------------------#
#hydra
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra.utils as hydra_utils
#wandb
import wandb
#dealing with files and paths etc.
import os
import itertools
import json
import joblib
from copy import deepcopy
#data handling
import pandas as pd
import numpy as np
#sklearn
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import root_mean_squared_error as rmse
#--------------------Functions--------------------#
#functions for converting search space from yaml to list of concrete model configs for sweeping
def expand_search_space_dim(space_dict):
    #conerting DictConfig to regular dict for easier handling and logging in wandb (fails downstream without it))
    space_dict = OmegaConf.to_container(space_dict, resolve=True)
    #prepare keys and values (variables) for expansion
    keys = list(space_dict.keys())#this gets names of all parameters (for example alpha, l1_ratio, etc) as a list
    values = []
    #Putting all values as lists (if not already lists) for easier expansion using itertools.product (which expects iterables for expansion, i.e. lists, not single values which might be the case)
    for value in space_dict.values():#iterating over values in different keys
        if isinstance(value, (list, tuple)):#if list or tuple, keep as list
            values.append(list(value))
        else:#if not make one-element list
            values.append([value])
    #prepare varaible to hold each configuration (combination of parameters) as a dict for easier handling and logging
    results = []
    #expanding using cartesian product of all values (all combinations of parameters) and creating dict for each combination with keys as parameter names and values as specific value for that parameter in that combination
    for combo in itertools.product(*values):
        #adding single configuration (combination of parameters) as a dict to results list
        results.append(dict(zip(keys, combo)))
    #returning output
    return results
def expand_search_space(search_cfg):
    #getting all pre-processing variants
    preprocess_runs = expand_search_space_dim(search_cfg.preprocess)
    #preparing list to hold all combinations of model and pre-processing configurations for sweeping
    runs = []
    #iterating through all models
    for model_name, model_space in search_cfg.models.items():
        #getting all variants for current model
        model_runs = expand_search_space_dim(model_space)
        #combining each model configuration with each pre-processing configuration and adding to runs list
        for model_params in model_runs:#iterating through all model params combinations for current model
            for preprocess_params in preprocess_runs:#iterating through all pre-processing params combinations
                #joining current model configuration with current pre-processing configuration and adding to runs list
                runs.append(
                    {
                        "model": {
                            "name": model_name,
                            "params": model_params,
                        },
                        "preprocess": preprocess_params,
                    }
                )
    #returning output
    return runs
#function to performing runf or single config
def evaluate_single_config(
    run_cfg: dict,
    X_train,
    y_train,
    names_num_cols_train,
    names_cat_cols_train,
    cfg: DictConfig,
):
    #importing code form modules (seprate files for better orgnization of logic)
    from pipeline import build_pipeline_cfg
    #building pipelien for current configuration (model + pre-processing)
    pipe = build_pipeline_cfg(
        names_num_cols_train,
        names_cat_cols_train,
        cfg_preprocess=run_cfg["preprocess"],
        cfg_model=run_cfg["model"],
    )
    #getting split for K-Fold (thanks to seed it would be the same for each run, so they can be compared)
    cv_splitter = KFold(
        n_splits=cfg.cv.n_splits,
        shuffle=cfg.cv.shuffle,
        random_state=cfg.seed,
    )
    #performing cross validation for current run(config)
    cv = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring=cfg.cv.scoring,
        return_train_score=True,
        n_jobs=-1,#use all availible cores
    )
    #retriving score results
    rmse_folds = -cv["test_score"]
    train_rmse_folds = -cv["train_score"]
    #transforming scores into usefull form
    cv_rmse_mean = float(np.mean(rmse_folds))
    cv_rmse_std = float(np.std(rmse_folds, ddof=1)) if len(rmse_folds) > 1 else 0.0
    cv_train_rmse_mean = float(np.mean(train_rmse_folds))
    #getting results into one (dict) variable
    result = {
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "cv_train_rmse_mean": cv_train_rmse_mean,
        "cv_rmse_folds": rmse_folds.tolist(),#NumPy array to python list fopr the convinience in logging
    }
    #returning output
    return result
#function for starting wandb run 
def start_wandb_run(cfg: DictConfig, run_cfg: dict, job_type: str):
    #starting wandb run for current configuration
    run = wandb.init(
        #setting up wandb info for run identification and organization in the dashboard
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group=cfg.logger.get("group", None),
        mode=cfg.logger.get("mode", "online"),
        job_type=job_type,
        #config part of wandb info
        config={
            "seed": cfg.seed,#no need for OmegaConf.to_container here since it's a single value, not a complex structure, but keeping consistent with other config logging
            "split": OmegaConf.to_container(cfg.split, resolve=True),#OmegaConf.to_container converts DictConfig to regular dict for better logging and handling in wandb, resolve=True resolves any references in the config (like ${...}) to their actual values
            "cv": OmegaConf.to_container(cfg.cv, resolve=True),
            "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
            "model": run_cfg["model"],#different extraction due to how run config is created
            "preprocess": run_cfg["preprocess"],#different extraction due to how run config is created
        },
        settings=wandb.Settings(
            init_timeout=cfg.logger.get("init_timeout", 180),
            x_disable_stats=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
        ),
        #for multiple runs in one python process as is the case in sweeping, to make sure each run is treated as a separate run in wandb and not as a continuation of previous run
        reinit="finish_previous",
    )
    #returning wandb run object
    return run#currently not used, but leaving for structure and possible future use
#--------------------Code--------------------#
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    #importing code form modules (seprate files for better orgnization of logic)
    from data import load_data_cfg
    from pipeline import build_pipeline_cfg
    #data loading
    y, x, names_num_cols, names_cat_cols = load_data_cfg(cfg_dataset = cfg.dataset)
    #splitting into train and test sets for cross validation results estimation (names of taken columns needs to be removed)
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg.split.test_size,
        random_state=cfg.seed,
    )
    names_num_cols_train = [col for col in names_num_cols if col in X_train.columns]
    names_cat_cols_train = [col for col in names_cat_cols if col in X_train.columns]            
    #getting config for each wandb run
    run_cfg = expand_search_space(cfg.search)
    #preparing variable to holde results
    sweep_results = []
    #going through all configurations
    for model_cfg in run_cfg:
        #srarting wandb run for current configuration
        start_wandb_run(cfg, model_cfg, job_type="sweep")
        #training/evaluating one config
        result = evaluate_single_config(
            run_cfg=model_cfg,#run specific config
            X_train=X_train,
            y_train=y_train,
            names_num_cols_train=names_num_cols_train,
            names_cat_cols_train=names_cat_cols_train,
            cfg=cfg,#global setting (stay the same for all runs)
        )
        #logging direct results of current run
        wandb.log(
            {
                "cv_rmse_mean": result["cv_rmse_mean"],
                "cv_rmse_std": result["cv_rmse_std"],
                "cv_train_rmse_mean": result["cv_train_rmse_mean"],
            }
        )
        #adding results of current run to program results (not wandb logging)
        sweep_results.append(
            {
                "model_cfg": deepcopy(model_cfg),
                "cv_rmse_mean": result["cv_rmse_mean"],
                "cv_rmse_std": result["cv_rmse_std"],
            }
        )
        #finishing current run after logging all needed info
        wandb.finish()
    #getting best configuration based on test rmse (the main metric for model selection) from all runs in sweep results
    best_result = min(sweep_results, key=lambda x: x["cv_rmse_mean"])
    best_model_cfg = best_result["model_cfg"]
    #making run for final model training and evaluation with best configuration (based on test rmse) on test set and logging it in wandb
    start_wandb_run(cfg, best_model_cfg, job_type="final")
    #creating final pipeline
    final_pipe = build_pipeline_cfg(
        names_num_cols_train,
        names_cat_cols_train,
        cfg_preprocess = best_model_cfg["preprocess"],
        cfg_model = best_model_cfg["model"],
    )
    #fitting of model
    final_pipe.fit(X_train, y_train)
    #getting prediction of model on test set
    y_pred = final_pipe.predict(X_test)#preproccesing taken care by pipeline
    #evaluating final model on test set
    final_test_rmse = float(rmse(y_test, y_pred))
    #logging results of currently run configuration from cv and current results on the test set in wandb
    wandb.log(
        {
            "best_cv_rmse_mean": best_result["cv_rmse_mean"],
            "best_cv_rmse_std": best_result["cv_rmse_std"],
            "final_test_rmse": final_test_rmse,
        }
    )
    #getting root folder for current hydra run
    run_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(run_dir, exist_ok=True)
    #getting paths for particular saves
    model_path = os.path.join(run_dir, "best_model.joblib")
    metrics_path = os.path.join(run_dir, "best_metrics.json")
    preds_path = os.path.join(run_dir, "test_predictions.csv")
    #saving pipeline
    joblib.dump(final_pipe, model_path)
    #saving metrics and configuration of the best model
    with open(metrics_path, "w") as file:#opens/creates json file for writing config and metrics of the best model
        json.dump(
            {
                "best_model_cfg": best_model_cfg,
                "best_cv_rmse_mean": best_result["cv_rmse_mean"],
                "best_cv_rmse_std": best_result["cv_rmse_std"],
                "final_test_rmse": final_test_rmse,
            },
            file,
            indent=2,
        )
    #saving predictions of the best model on the test set as csv file (if possible)
    try:
        #creating dataframe with true and predicted values for better analysis and saving as csv file
        pd.DataFrame(
            {
                "y_true": np.asarray(y_test),
                "y_pred": np.asarray(y_pred),
            }
        ).to_csv(preds_path, index=False)
    except Exception:
        pass
    #logging results of best models as artifact to wandb
    artifact = wandb.Artifact("best-model", type="model")
    artifact.add_file(model_path)
    artifact.add_file(metrics_path)
    if os.path.exists(preds_path):
        artifact.add_file(preds_path)
    wandb.log_artifact(artifact)
    #finishing run of best model
    wandb.finish()

if __name__ == "__main__":
    main()