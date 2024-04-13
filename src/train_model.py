# train_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, f1_score
import joblib  
import json
def load_processed_data(filepath):
    """Load preprocessed data from a CSV file."""
    return pd.read_csv(filepath)

def train_model(X, y):
    """Train the model using the loaded hyperparameters."""
    model = xgb.XGBClassifier(random_state=42)
    
    """Load hyperparameters from a JSON file."""
    path = './configuration/hyperparameters.json'
    
    with open(path, 'r') as file:
        try:
            param_space = json.load(file)
        except json.JSONDecodeError:
            print("Error decoding JSON. Please check the format of the file.")
            raise
        except FileNotFoundError:
            print("Hyperparameter file not found. Please check the file path.")
            raise
    
    # Setup F1 micro scorer
    f1_micro_scorer = make_scorer(f1_score, average='micro')
    # perform hyperparameter tuning with RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_space, scoring=f1_micro_scorer, cv=5, n_jobs=-1, verbose=1, random_state=42)
    random_search.fit(X, y)
    return random_search.best_estimator_, random_search.best_score_

def evaluate_model(model, X, y):
    # predict and evaluate
    predictions = model.predict(X)
    f1 = f1_score(y, predictions, average='micro')
    return f1

if __name__ == "__main__":
    # load data
    X = load_processed_data('../data/processed/train_processed.csv')
    y = pd.read_csv('../data/raw/train_labels.csv', index_col='building_id').squeeze() - 1

    # Split into training and validation sets to evaluate performance
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model, best_f1_score = train_model(X_train, y_train)
    joblib.dump(best_model, '../models/best_model.pkl')

    # Print the best F1 score obtained during cross-validation
    print(f"Best F1 Score from CV: {best_f1_score:.4f}")

    # Evaluate model on the validation set and print the result
    val_f1_score = evaluate_model(best_model, X_val, y_val)
    print(f"Validation F1 Score: {val_f1_score:.4f}")
