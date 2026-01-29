print("Importing packages...")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import loguniform, randint
from sklearn.metrics import accuracy_score, mean_absolute_error
from utils.updateStats import getStats, updateStats, createStats
pd.set_option('display.max_columns', None)

def train_eval_xgboost(dataset, params, update_stats_param, filter_num, challengers, best_score=-np.inf, MODEL_NAME="model"):
    training_results = {}
    training_results.update(params)
    training_results.update(update_stats_param)
    training_results["filter_num"] = filter_num
    training_results["challengers"] = challengers
    training_results["MODEL_NAME"] = MODEL_NAME
    
    ############################################################################################################################################
    ##### HYPER-PARAMETERS #####
    ############################################################################################################################################
    print("\nPrinting hyperparameters")
    print(f"filter_num={filter_num}")
    print(f"challengers={challengers}")
    print(f"params={params}")
    print(f"update_stats_param={update_stats_param}")

    ############################################################################################################################################
    ##### CREATING THE DATASET #####
    ############################################################################################################################################
    print("\nCreating the training dataset...")
    allData = pd.read_csv(dataset)

    if challengers:
        allDataNoChallengers = allData
    else:
        allDataNoChallengers = allData[allData["tourney_level"].astype(str).isin(["A", "G", "M", "F", "D", "O"])]

    ###### Create Dataset ######
    clean_data = allDataNoChallengers[~allDataNoChallengers["tourney_date"].astype(str).str.contains("2025")]
    clean_data = clean_data.reset_index(drop=True)

    final_dataset = []
    prev_stats = createStats()

    # Iterate through each row in clean_data
    for index, row in tqdm(clean_data.iterrows(), total=len(clean_data)):
        player1 = {
            "ID": row["p1_id"],
            "ATP_RANK": row["p1_rank"],
            "AGE": row["p1_age"],
            "HEIGHT": row["p1_ht"],
        }

        player2 = {
            "ID": row["p2_id"],
            "ATP_RANK": row["p2_rank"],
            "AGE": row["p2_age"],
            "HEIGHT": row["p2_ht"],
        }

        match = {
            "BEST_OF": row["best_of"],
            "DRAW_SIZE": row["draw_size"],
            "SURFACE": row["surface"],
            "ROUND": row["round"],
        }

        ########## GET STATS ##########
        # Call getStatsPlayers function
        output = getStats(player1, player2, match, prev_stats)

        # Append sorted stats to final dataset
        match_data = dict(sorted(output.items()))
        match_data["RESULT"] = row.RESULT
        final_dataset.append(match_data)

        ########## UPDATE STATS ##########
        prev_stats = updateStats(row, prev_stats, **update_stats_param)

    # Convert final dataset to DataFrame
    final_dataset = pd.DataFrame(final_dataset)

    final_dataset = final_dataset[final_dataset.index > filter_num]

    ############################################################################################################################################
    ###### TRAINING #######
    ############################################################################################################################################
    print("\nStarting training...")
    # Convert data to numpy
    data = final_dataset.to_numpy(dtype=object)
    np.random.shuffle(data)

    # Split the data using an 95% split between training and testing
    split = 0.95
    value = round(split*len(data))

    data_train = data[:value,:]
    data_test = data[value:,:]

    print("Training Data: "+str(data_train.shape))
    print("Testing Data: "+str(data_test.shape))

    # Define several mappers
    mapper = np.vectorize(lambda x: "Player 2 Wins" if x == 0 else "Player 1 Wins")
    reverse_mapper = np.vectorize(lambda x: 0 if x == "Player 2 Wins" else 1)

    # Training data
    x_train = data_train[:,:-1]
    y_pred_train = mapper(data_train[:,-1:]).squeeze()

    # Testing data
    x_test = data_test[:,:-1]
    y_pred_test = mapper(data_test[:,-1:]).squeeze()

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        **params
    )

    # Fit using training data and early stopping on validation data
    model.fit(x_train, reverse_mapper(y_pred_train))

    # Make predictions
    predictions_train = model.predict(x_train)
    predictions_test = model.predict(x_test)

    # Calculate accuracy
    train_accuracy = accuracy_score(reverse_mapper(y_pred_train), predictions_train)
    test_accuracy = accuracy_score(reverse_mapper(y_pred_test), predictions_test)
    print("Train Accuracy: " + str(train_accuracy))
    print("Test Accuracy: " + str(test_accuracy))
    
    training_results["train_accuracy"] = train_accuracy
    training_results["test_accuracy"] = test_accuracy

    ############################################################################################################################################
    ###### EVALUTAION #######
    ############################################################################################################################################
    print("\nStarting evaluation...")

    def predict_twice_average(player1: dict, player2: dict, match: dict, xgb_model, prev_stats):
        """
        Returns the probability of player 1 winning
        """
        p1_prob = []
        p2_prob = []

        # Call getStatsPlayers function
        output = getStats(player1, player2, match, prev_stats)
        match_data = pd.DataFrame([dict(sorted(output.items()))])
        probs = xgb_model.predict_proba(np.array(match_data, dtype=object))[:, ::-1]

        p1_prob.append(probs[0][0])
        p2_prob.append(probs[0][1])

        output = getStats(player2, player1, match, prev_stats)
        match_data = pd.DataFrame([dict(sorted(output.items()))])
        probs = xgb_model.predict_proba(np.array(match_data, dtype=object))[:, ::-1]

        p1_prob.append(probs[0][1])
        p2_prob.append(probs[0][0])

        return round(float(np.mean(p1_prob)), 4)

    def run_evaluation(xgb_model, evaulation_data, prev_stats):
        predictions = []
        elo_predictions = []
        probabilities = []
        results = []
        counter = 0

        evaulation_data = evaulation_data[evaulation_data["tourney_date"].astype(str).str.contains("2025")]
        for index, row in tqdm(evaulation_data.iterrows(), total=len(evaulation_data)):
            player1 = {
                "ID": row["p1_id"],
                "ATP_RANK": row["p1_rank"],
                "AGE": row["p1_age"],
                "HEIGHT": row["p1_ht"],
            }

            player2 = {
                "ID": row["p2_id"],
                "ATP_RANK": row["p2_rank"],
                "AGE": row["p2_age"],
                "HEIGHT": row["p2_ht"],
            }

            match = {
                "BEST_OF": row["best_of"],
                "DRAW_SIZE": row["draw_size"],
                "SURFACE": row["surface"],
                "ROUND": row["round"],
            }

            ########## PREDICT ##########
            if row["tourney_level"] in ["A", "G", "M", "F", "D", "O"] and row["round"] not in ["Q1", "Q2", "Q3"]:
                # Baseline accuracy
                if prev_stats["elo_players"][row["p1_id"]] >= prev_stats["elo_players"][row["p2_id"]]:
                    elo_predictions.append(1)
                else:
                    elo_predictions.append(0)
                
                prob_prediction = predict_twice_average(player1, player2, match, xgb_model, prev_stats)
                predictions.append(1) if prob_prediction >= 0.5 else predictions.append(0)
                probabilities.append(prob_prediction)
                counter += 1
                
                # Save result to compare
                results.append(row["RESULT"])

            # Update the stats of the match after it has been predicted!
            prev_stats = updateStats(row, prev_stats, **update_stats_param)

        print (
            f"EVALUATION RESULTS:\n"
            f"Evaluated {counter} matches...\n"
            f"Baseline ELO Accuracy: {accuracy_score(elo_predictions, results)}\n"
            f"Accuracy Score: {accuracy_score(predictions, results)}\n"
            f"MAE: {mean_absolute_error(predictions, results)}"
        )
        
        return accuracy_score(predictions, results), mean_absolute_error(predictions, results)

    # Run your custom sequential evaluation on 2025 validation set
    allData_upto_2025 = allData[~allData["tourney_date"].astype(str).str.contains("2025")]

    prev_stats_eval = createStats()

    # Update up until end of 2024
    for index, row in tqdm(allData_upto_2025.iterrows(), total=len(allData_upto_2025)):
        ########## UPDATE STATS ##########
        prev_stats_eval = updateStats(row, prev_stats_eval, **update_stats_param)

    score, mae = run_evaluation(xgb_model=model, evaulation_data=allData, prev_stats=prev_stats_eval)
    
    training_results["score2025"] = score
    training_results["mae2025"] = mae
    
    # Save model
    if score > best_score:
        model.save_model(f"./models/{MODEL_NAME}.json")
    
    return training_results

if __name__ == '__main__':
    params = {
        "n_estimators": 250,                                #randint(10, 500),
        "learning_rate": 0.04,                              #loguniform(1e-3, 3e-1),
        "max_depth": 5,                                     #randint(3, 15),
        "subsample": 0.9,                                   #loguniform(0.5, 1.0),
        "colsample_bytree": 0.95,                           #loguniform(0.5, 1.0),
        "gamma": 0.2,                                       #loguniform(1e-3, 1.0),
        "reg_alpha": 0.5,                                   #loguniform(1e-4, 10.0),
        "reg_lambda": 5,                                    #loguniform(1e-2, 50.0),
    }

    update_stats_param = {
        "k_factor": None, 
        "base_k_factor": 43, 
        "max_k_factor": 62, 
        "div_number": 800, 
        "bonus_after_layoff": True
    }
    
    train_eval_xgboost("./data/0cleanDatasetWithQualifiersWith2025.csv", 
                       params,
                       update_stats_param,
                       filter_num=10000, 
                       challengers=False,
                       best_score=np.inf, # doesn't save the model
                       MODEL_NAME="testModelXGBoost")