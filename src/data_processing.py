# data_preprocessing.py
import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Example: One-hot encoding, filling NaNs, type conversions
    df_processed = pd.get_dummies(df)
    df_processed.fillna(0, inplace=True)
    return df_processed

if __name__ == "__main__":
    train_data = load_data('../data/raw/train_values.csv')
    train_labels = load_data('../data/raw/train_labels.csv')
    test_data = load_data('../data/raw/test_values.csv')

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    train_data.to_csv('../data/processed/train_processed.csv', index=False)
    test_data.to_csv('../data/processed/test_processed.csv', index=False)
