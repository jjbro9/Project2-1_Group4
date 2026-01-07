from io import StringIO

import pandas as pd
import requests
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# training the model based on the hyperparams
def make_model():
    df = pd.read_csv('experiments_merged.csv')
    print(df.columns)

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
            df[col] = df[col].replace({'True': 1, 'False': 0}).astype(int)

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

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=1
    # )
    print(X.dtypes)

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=1,
        verbose=1
    )

    kfold = KFold(n_splits=10, shuffle=True, random_state=1)

    cv_score = cross_val_score(gb_model, X, y, cv=kfold, scoring='r2')
    cv_mse = -cross_val_score(gb_model, X, y, cv=kfold,
                              scoring='neg_mean_squared_error')
    cv_mae = -cross_val_score(gb_model, X, y, cv=kfold,
                              scoring='neg_mean_absolute_error')

    # predictions = gb_model.predict(X)
    # rmse = mean_squared_error(y, predictions)
    # print("R²:", r2_score(y, predictions))
    # print("RMSE:", rmse)
    # print("MAE:", mean_absolute_error(y, predictions))

    print("K-Fold results")
    print(f"R² scores: {cv_score}")
    print(f"mean R²: {cv_score.mean():.4f} (+/- {cv_score.std():.4f})")

    print(f"MSE scores: {cv_mse}")
    print(f"mean MSE: {cv_mse.mean():.4f} (+/- {cv_mse.std():.4f})")

    print(f"MAE scores: {cv_mae}")
    print(f"mean MAE: {cv_mae.mean():.4f} (+/- {cv_mae.std():.4f})")

    gb_model.fit(X, y)

    print("Feature Importances")
    feature_importances = gb_model.feature_importances_
    for name, score in zip(X.columns, feature_importances):
        print(f"{name}: {score:.4f}")

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
    predictions = model.predict(data)
    for i, prediction in enumerate(predictions):
        print(f"Row {i}: Predicted Reward = {prediction:.4f}")


model, feature_names = make_model()
while True:
    url = input("\nEnter the URL of the CSV file (or 'quit' to exit): ").strip()
    if url.lower() == 'quit':
        break
    predict(model, feature_names, url)
