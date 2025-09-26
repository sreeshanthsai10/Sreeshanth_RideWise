# src/utils.py
import pandas as pd
import numpy as np

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure datetime if present
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'])
    # cyclical hour features
    df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
    df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
    # combined features
    df['temp_feels_like'] = (df['temp'] + df['atemp']) / 2
    df['weather_comfort'] = df['atemp'] * (1 - df['hum'])
    # rush hour flags and weekend
    df['is_rush_hour'] = (((df['hr'] >= 7) & (df['hr'] <= 9)) | ((df['hr'] >= 17) & (df['hr'] <= 19))).astype(int)
    df['is_weekend'] = df['weekday'].isin([0, 6]).astype(int)
    return df

def add_lag_features(df: pd.DataFrame, target_col='cnt') -> pd.DataFrame:
    df = df.copy()
    # make sure sorted by date/hour
    if 'dteday' in df.columns:
        df = df.sort_values(['dteday', 'hr'])
    else:
        df = df.sort_values(['hr'])
    df['prev_day_same_hour'] = df.groupby('hr')[target_col].shift(24)
    # fallback: median of target
    df['prev_day_same_hour'] = df['prev_day_same_hour'].fillna(df[target_col].median() if target_col in df.columns else 200)
    return df

def build_feature_matrix(df: pd.DataFrame):
    """Return X_encoded, y, and the encoded dataframe columns (useful to persist)"""
    df = df.copy()
    X = df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
            'temp', 'atemp', 'hum', 'windspeed', 'hr',
            'hr_sin', 'hr_cos', 'temp_feels_like', 'weather_comfort',
            'is_rush_hour', 'is_weekend', 'prev_day_same_hour']]
    y = df['cnt'] if 'cnt' in df.columns else None
    X_encoded = pd.get_dummies(X, columns=['season', 'weathersit'], drop_first=True)
    return X_encoded, y, X_encoded.columns

def prepare_input_df(sample_input: dict, feature_columns):
    """Turn a dict (single sample) into model-ready DataFrame with same columns as training."""
    input_df = pd.DataFrame([sample_input.copy()])
    # apply feature engineering
    input_df = feature_engineer(input_df)
    # ensure prev_day_same_hour provided or fallback
    if 'prev_day_same_hour' not in input_df.columns:
        input_df['prev_day_same_hour'] = sample_input.get('prev_day_same_hour', 200)
    # select same feature order as training
    input_features = input_df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
                               'temp', 'atemp', 'hum', 'windspeed', 'hr',
                               'hr_sin', 'hr_cos', 'temp_feels_like', 'weather_comfort',
                               'is_rush_hour', 'is_weekend', 'prev_day_same_hour']]
    input_encoded = pd.get_dummies(input_features, columns=['season', 'weathersit'], drop_first=True)
    # reindex to training columns
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    return input_encoded
