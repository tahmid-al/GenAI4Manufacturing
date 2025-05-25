# Preprocessing script for NASA C-MAPSS data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(train_path, test_path, rul_path):
    # Load text data
    train_df = pd.read_csv(train_path, sep='\s+', header=None)
    test_df = pd.read_csv(test_path, sep='\s+', header=None)
    rul_df = pd.read_csv(rul_path, sep='\s+', header=None)
    return train_df, test_df, rul_df

def add_column_names(df):
    cols = ['unit', 'cycle'] + \
           [f'os_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]
    df.columns = cols
    return df

def calculate_rul(df):
    rul_df = df.groupby('unit')['cycle'].max().reset_index()
    rul_df.columns = ['unit', 'max_cycle']
    df = df.merge(rul_df, on='unit')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

def normalize(df, scaler=None):
    feat_cols = [col for col in df.columns if col not in ['unit', 'cycle', 'RUL']]
    if scaler is None:
        scaler = MinMaxScaler()
        df[feat_cols] = scaler.fit_transform(df[feat_cols])
        return df, scaler
    else:
        df[feat_cols] = scaler.transform(df[feat_cols])
        return df, scaler

def windowize(df, seq_length=30):
    X, y = [], []
    units = df['unit'].unique()
    feat_cols = [col for col in df.columns if col not in ['unit', 'cycle', 'RUL']]
    for unit in units:
        unit_df = df[df['unit'] == unit]
        for i in range(len(unit_df) - seq_length + 1):
            X.append(unit_df.iloc[i:i+seq_length][feat_cols].values)
            y.append(unit_df.iloc[i+seq_length-1]['RUL'])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    train_path = "../data/train_FD001.txt"
    test_path = "../data/test_FD001.txt"
    rul_path = "../data/RUL_FD001.txt"

    train_df, test_df, rul_df = load_data(train_path, test_path, rul_path)
    train_df = add_column_names(train_df)
    test_df = add_column_names(test_df)
    
    train_df = calculate_rul(train_df)

    # Normalize
    train_df, scaler = normalize(train_df)
    test_df, _ = normalize(test_df, scaler=scaler)

    # Save normalized for later use
    train_df.to_csv("../data/train_FD001_normalized.csv", index=False)
    test_df.to_csv("../data/test_FD001_normalized.csv", index=False)
    
    # Create LSTM windows
    X_train, y_train = windowize(train_df, seq_length=30)
    np.save("../data/X_train_FD001.npy", X_train)
    np.save("../data/y_train_FD001.npy", y_train)

    print("Preprocessing done! Shapes:", X_train.shape, y_train.shape)