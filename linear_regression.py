from io import StringIO

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# training the model based on the hyperparams
def make_model():
    df = pd.read_csv('experiments-merged-clean.csv')
    print(df.columns)

    drop_cols = ['run_id', 'timestamp', 'user', 'env_path',
                 'platform', 'gpu_name', 'gpu_mem_gb', 'ram_gb', 'summary_freq', 'cpu_count']
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

    print(X.dtypes)

    # scale features for better coeffient view
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    lr_model = LinearRegression()

    # Apply the kfold and cross validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=None)

    cv_score = cross_val_score(lr_model, X_scaled, y, cv=kfold, scoring='r2')
    cv_mse = -cross_val_score(lr_model, X_scaled, y, cv=kfold,
                              scoring='neg_mean_squared_error')
    cv_mae = -cross_val_score(lr_model, X_scaled, y, cv=kfold,
                              scoring='neg_mean_absolute_error')

    # Printing the results
    print("\nK-Fold results")
    print(f"R² scores: \t{cv_score}")
    print(f"Mean R²: \t{cv_score.mean():.4f} (+/- {cv_score.std():.4f})")

    print(f"MSE scores: \t{cv_mse}")
    print(f"Mean MSE: \t {cv_mse.mean():.4f} (+/- {cv_mse.std():.4f})")

    print(f"MAE scores: \t{cv_mae}")
    print(f"Mean MAE: \t{cv_mae.mean():.4f} (+/- {cv_mae.std():.4f})")

    lr_model.fit(X_scaled, y)

    # Sorting the results
    print("\nCoefficients (sorted)")
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr_model.coef_,
        'Abs_Coefficient': np.abs(lr_model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)

    for index, row in coefficients.iterrows():
        print(f"{row['Feature']}: {row['Coefficient']:.4f}")

    print(f"\nIntercept: {lr_model.intercept_:.4f}")

    return lr_model, list(X.columns), scaler

# Loading the url


def load(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    except:
        return None

# Predicting the results


def predict(model, feature_names, scaler, url):
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

    scaled_data = scaler.transform(data)

    predictions = model.predict(scaled_data)
    for i, prediction in enumerate(predictions):
        print(f"Row {i}: Predicted Reward = {prediction:.4f}")


model, feature_names, scaler = make_model()

while True:
    url = input("\nEnter the URL of the CSV file (or 'quit' to exit): ").strip()
    if url.lower() == 'quit':
        break
    predict(model, feature_names, scaler, url)
