from io import StringIO

import numpy as np
import pandas as pd
import requests
from numpy.ma.extras import average
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


# training the model based on the hyperparams
def make_model():
    df = pd.read_csv('experiments-merged-clean.csv')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    drop_cols = ['run_id', 'timestamp', 'user', 'env_path',
                 'platform', 'gpu_name', 'gpu_mem_gb', 'ram_gb']
    df = df.drop(columns=drop_cols, errors='ignore')

    categorical_cols = ['algorithm', 'behavior_name', 'learning_rate_schedule']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    bool_cols = ['normalize']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'True': 1, 'False': 0})

    for col in df.columns:
        if col != 'mean_reward':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(axis=1, how='all')

    target_col = 'mean_reward'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    assert not X.isna().any().any(), "NaNs remain in X!"
    assert not y.isna().any(), "NaNs remain in y!"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    n_col = len(df)

    param_grid = {
        "learning_rate": [.001, .002, .003, .004, .005, .006, .007, .008, .009, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1, .15, .2, .25, .3, .4, .5, .6],
    }

    gb_model = GradientBoostingRegressor(
        n_estimators=100000,
        learning_rate=0.022,
        max_depth=12,
        min_samples_split=105,
        min_samples_leaf=42,
        subsample=0.9,
        random_state=1,
        n_iter_no_change=20,
        validation_fraction=.1
    )

    optimize(gb_model, param_grid, X_train, y_train, 20)

    return gb_model, list(X.columns)


def load(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    except:
        return None


def optimize(gb_model, param_grid, X_train, y_train, cv):
    grid_search = GridSearchCV(
        estimator=gb_model,
        param_grid=param_grid,
        scoring='r2',
        cv=cv,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best Tree Complexity Score (R2): {grid_search.best_score_}")
    print(f"Best Parameters: {grid_search.best_params_}")

# predicting a new data set, given a csv file


def predict(model, feature_names, url):
    data = load(url)
    if data is None:
        print("no data")
        return None

    categorical_cols = ['algorithm', 'behavior_name']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0

    data = data[feature_names]
    data = data.fillna(data.mean())


model, feature_names = make_model()
while True:
    url = input("\nEnter the URL of the CSV file (or 'quit' to exit): ").strip()
    if url.lower() == 'quit':
        break
    predictions = predict(model, feature_names, url)
