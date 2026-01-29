# sweep_train.py
import os
import pandas as pd
import numpy as np
from scipy.stats import loguniform, randint
from train import train_eval_xgboost

N_ITER = 2000
DATASET = "./data/0cleanDatasetWithQualifiersWith2025.csv"
FILTER_NUM = 10000
CHALLENGERS = False
LOG_PATH = "hyperparameters/data/final_xgb_param_log_test.csv"

update_stats_param = {
    "k_factor": None,
    "base_k_factor": 43,
    "max_k_factor": 62,
    "div_number": 800,
    "bonus_after_layoff": False,
}

# Random search space
param_dist = {
    "n_estimators": randint(50, 450),
    "learning_rate": loguniform(1e-3, 0.10),
    "max_depth": randint(3, 10),
    "subsample": loguniform(0.8, 1.0),
    "colsample_bytree": loguniform(0.8, 1.0),
    "gamma": loguniform(1e-2, 0.8),
    "reg_alpha": loguniform(1e-4, 8),
    "reg_lambda": loguniform(1e-2, 50.0),
}

if __name__ == "__main__":
    header_written = os.path.isfile(LOG_PATH)
    
    best_score = -np.inf
    best_params = None

    for i in range(1, N_ITER + 1):
        # Sample a random parameter set
        params = {k: v.rvs() for k, v in param_dist.items()}

        # Train & evaluate via your train.py function
        res = train_eval_xgboost(
            dataset=DATASET,
            params=params,
            update_stats_param=update_stats_param,
            filter_num=FILTER_NUM,
            challengers=CHALLENGERS,
            best_score=best_score,
            MODEL_NAME=f"best_final_xgb_model",
        )
        
        score = res["score2025"]

        # Log results (flat dict already includes hyperparams + metrics)
        row = {"iteration": i, **res}
        pd.DataFrame([row]).to_csv(
            LOG_PATH, mode="a", header=not header_written, index=False
        )
        header_written = True
        
        if score > best_score:
            best_score = score
            best_params = row

        print(f"[{i}/{N_ITER}] score2025={res.get('score2025'):.4f} "
              f"train_acc={res.get('train_accuracy'):.4f} "
              f"test_acc={res.get('test_accuracy'):.4f}")
