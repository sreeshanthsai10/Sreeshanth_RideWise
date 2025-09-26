# src/train_model.py
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import utils

def main(data_path, model_out, cols_out, random_state=42):
    print("Loading data:", data_path)
    df = pd.read_csv(data_path)
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'])

    # feature engineering
    df = utils.feature_engineer(df)
    df = utils.add_lag_features(df, target_col='cnt')

    # build X/y and encode
    X_encoded, y, feature_columns = utils.build_feature_matrix(df)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=random_state)

    # hyperparameter distribution
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.8],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rs = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3,
                            scoring='r2', n_jobs=-1, random_state=random_state, verbose=2)
    print("Starting RandomizedSearchCV...")
    rs.fit(X_train, y_train)
    best_model = rs.best_estimator_
    print("Best params:", rs.best_params_)

    # evaluation
    preds = best_model.predict(X_test)
    print("R2:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))

    # save model and feature columns
    joblib.dump(best_model, model_out)
    joblib.dump(feature_columns, cols_out)
    print("Saved model to", model_out)
    print("Saved feature columns to", cols_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data/hour.csv", help="Path to CSV dataset")
    parser.add_argument("--model_out", default="../models/bike_model.pkl", help="Path to save trained model")
    parser.add_argument("--cols_out", default="../models/feature_columns.pkl", help="Path to save feature columns")
    args = parser.parse_args()
    main(args.data_path, args.model_out, args.cols_out)
