# Train, validate and test Software Vulnerability Prediction models
import os
import datetime
import time
from pathlib import Path

import pandas as pd
import numpy as np
import pickle

from scipy import sparse
from scipy.sparse import coo_matrix

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, make_scorer, \
    classification_report


# ANSI escape sequences used for decorating console output.\n",
bold = '\033[1m'
bold_end = '\033[0m'


metrics_file = ""
report_file = ""
metrics_headers = ['metric', 'project', 'model', 'features', 'num_predictions', 'percent_predictions', 'recall',
                   'precision', 'f1', 'mcc', 'train_time', 'pred_time']

# Paths.
models_path = ""
models_sub_path = ""

# For repeated testing
random_state = 42

# Create scorers.
mcc_scorer = make_scorer(matthews_corrcoef)


def load_extracted_features(project_name, nlp_dir: str):
    return sparse.load_npz(f"{nlp_dir}/{project_name}_features_sparse.npz")


def tune_hyper_parameters_with_grid_search(model, param_grid, X, y, n_jobs: int):
    # Create the grid search cv object.
    grid_search = GridSearchCV(
        verbose=4,
        estimator=model,
        param_grid=param_grid,
        n_jobs=n_jobs,
        cv=3,
        scoring=mcc_scorer)

    # Perform grid search and cross validate.
    grid_search_result = grid_search.fit(X, y)

    # Return results.
    return grid_search_result


def tune_train_and_validate(model_name, pipeline, param_grid, X, y, n_jobs: int, is_save_model=True):
    print("Processing % s:" % model_name)

    # Perform gridsearch.
    print("\tPerforming grid search cross validation...")
    grid_search_results = tune_hyper_parameters_with_grid_search(pipeline, param_grid, X, y, n_jobs=n_jobs)
    print("\tFinished grid search cross validation.")

    # Print results.
    print("\tGrid search results:")
    print_grid_search_cv_results(grid_search_results, "\t\t")

    # Get the best model.
    best_model = grid_search_results.best_estimator_
    print(f"Best: {best_model}")
    # Train the best model with the entire training set.
    print("\tTraining model with best hyper parameters...")
    best_model.fit(X, y)
    print("\tFinished training.")

    # Save best model.
    if is_save_model:
        save_model(best_model, active_project_name, model_name)
        print("\tSaved %s model." % model_name)

    return best_model


def print_grid_search_cv_results(grid_search_results, row_prefix):
    cv_results = grid_search_results.cv_results_
    #     print(cv_results)
    for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
        print(row_prefix, mean_score, params)


def save_model(model, project_name, model_name):
    if not os.path.exists(models_path + "/" + project_name):
        os.makedirs(models_path + "/" + project_name)
    pickle.dump(model, open(f"{models_path}/{project_name}/{model_name}.model", "wb"))


def load_model(project_name, model_name):
    return pickle.load(open(models_path + "/" + project_name + "/" + model_name + ".model", 'rb'))


def evaluate_model(model_prefix, model_pipeline, model_param_grid, X_train, y_train, X_test, y_test, n_jobs: int,
                   load=False):
    # Train and predict
    t_start = time.time()
    if load:
        model_best = load_model(active_project_name, model_prefix)
    else:
        model_best = tune_train_and_validate(model_prefix, model_pipeline, model_param_grid, X_train, y_train,
                                             is_save_model=True, n_jobs=n_jobs)
    train_time = time.time() - t_start
    p_start = time.time()
    y_predict = model_best.predict(X_test)
    report = classification_report(y_test, y_predict, output_dict=True)
    print(report)
    report = {f"{model_prefix}_{k}": v for k, v in report.items()}
    df = pd.DataFrame(report).transpose()
    df.to_csv(report_file, mode='a')

    pred_time = time.time() - p_start

    num_predictions = len([x for x in y_predict if x])
    num_predictions_p = num_predictions / X_test.shape[0]

    weighting = 'binary'
    mcc = matthews_corrcoef(y_test, y_predict)
    f1 = f1_score(y_test, y_predict, average=weighting)
    recall = recall_score(y_test, y_predict, average=weighting)
    precision = precision_score(y_test, y_predict, average=weighting)
    print(mcc, f1, recall, precision)

    row = ['effectiveness', active_project_name, model_prefix, models_sub_path, num_predictions, num_predictions_p,
           recall, precision, f1, mcc, train_time, pred_time]
    df = pd.DataFrame(row, metrics_headers)
    df.to_csv(metrics_file, mode='a', header=False)

    return y_predict


def record_predictions(data, y_predict, model_prefix, predicted_path: str):
    X_train, X_test, y_train, y_test = train_test_split(data, data, train_size=0.8, random_state=random_state)

    y_test = y_test.drop(columns=['commit_time', 'commit_time_ordinal'])
    y_test['predicted'] = y_predict
    y_test.to_csv(f"{predicted_path}/{active_project_name}_{model_prefix}_{models_sub_path}_{random_state}.csv",
                  index=False)


def train_test(data, model_name: str, model_out_path: str, results_path: str, project_name: str,
               nlp_features, n_jobs: int = 2):
    global metrics_file, report_file

    metrics_file = f"{results_path}/metrics.csv"
    report_file = f"{results_path}/{model_name.lower()}_report.csv"
    # Open a results file
    if not os.path.isfile(metrics_file):
        rfile = open(metrics_file, 'a')
        rfile.write(
            'metric,project,model,features,num_predictions,percent_predictions,recall,precision,f1,mcc,train_time,pred_time\n')

    # Open a results file
    if not os.path.isfile(report_file):
        rfile = open(report_file, 'a')

    print("Model type: %s" % model_name)

    global active_project_name, model_param_grid, model_prefix, model_pipeline
    active_project_name = project_name

    print("Project name: %s" % active_project_name)

    # Set models path after data grouping options set.
    global models_sub_path
    models_sub_path = "_nlp"

    global models_path
    models_path = f"{model_out_path}/{models_sub_path[1:]}"
    print("Models path: %s" % models_path)

    # Extract labels.
    y = data["target"]

    # Create empty input.
    X = coo_matrix((y.shape[0], 0))
    print(X.shape)

    if nlp_features is not None:
        print("Adding NLP features.")
        # Add to X.
        X = sparse.hstack([X, nlp_features])
        print(X.shape)

    # Sort Data
    if "commit_time" in data:
        data['commit_time'] = pd.to_datetime(data['commit_time'])
        data["commit_time_ordinal"] = data["commit_time"].apply(lambda x: x.toordinal())

        # Sort X.
        X_dense = X.toarray()
        sorted_indices = np.argsort(data["commit_time_ordinal"])
        X_dense_sorted = X_dense[sorted_indices]
        X = sparse.csr_matrix(X_dense_sorted)

        # Sort y.
        data = data.sort_values(by='commit_time_ordinal', ascending=True)
        y = data["target"]
        y = y.reset_index(drop=True)
    else:
        data['commit_time'] = datetime.datetime.now()
        data['commit_time_ordinal'] = 0

    # Simple train, test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random_state)

    print(X_train.shape)
    print(X_test.shape)

    # Create models.
    if model_name == "KNN":
        # KNN pipeline.
        model = KNeighborsClassifier(n_jobs=n_jobs)
        model_pipeline = make_pipeline(model)
        model_prefix = model.__class__.__name__.lower()

        # Parameter grid.
        model_param_grid = [
            {
                model_prefix + '__weights': ['uniform', 'distance'],
                model_prefix + '__n_neighbors': [2, 3, 4, 5, 10, 20]
            }
        ]
    elif model_name == "SVC":
        # SVC pipeline.
        model = SVC(max_iter=200)
        model_pipeline = make_pipeline(model)
        model_prefix = model.__class__.__name__.lower()

        # Parameter grid.
        model_param_grid = [
            {
                model_prefix + '__C': [0.01, 0.1, 1, 10],
                model_prefix + '__kernel': ['linear', 'rbf']
            }
        ]
    elif model_name == "RFC":
        # Random forest pipeline.
        model = RandomForestClassifier(n_jobs=n_jobs)
        model_pipeline = make_pipeline(model)
        model_prefix = model.__class__.__name__.lower()

        # Parameter grid.
        model_param_grid = [
            {
                model_prefix + '__max_depth': [10, 100, None],
                model_prefix + '__min_samples_leaf': [1, 2, 4],
                model_prefix + '__n_estimators': [200, 600, 1000]
            }
        ]
    elif model_name == "Adaboost":
        # AdaBoost pipeline.
        model = AdaBoostClassifier()
        model_pipeline = make_pipeline(model)
        model_prefix = model.__class__.__name__.lower()

        # Parameter grid.
        model_param_grid = [
            {
                model_prefix + '__n_estimators': [30, 50, 100],
                model_prefix + '__learning_rate': [0.01, 0.1, 1, 10]
            }
        ]
    print(f"Param grid {model_param_grid}")
    # Tune, train and validate model.
    y_predict = evaluate_model(model_prefix, model_pipeline, model_param_grid, X_train, y_train, X_test, y_test,
                               load=False, n_jobs=n_jobs)
    predicted_path = Path(results_path, "predicted")

    if not predicted_path.exists():
        predicted_path.mkdir(parents=True)

    record_predictions(data, y_predict, model_prefix, predicted_path=str(predicted_path))
