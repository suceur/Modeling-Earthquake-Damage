# predict_model.py
import pandas as pd
import joblib

def load_model(path):
    """Load a trained model from a file."""
    return joblib.load(path)

def make_predictions(model, X):
    """Use the loaded model to make predictions."""
    return model.predict(X)

if __name__ == "__main__":
    # Load the trained model
    model = load_model('../models/best_model.pkl')
    # Load test data
    test_data = pd.read_csv('../data/processed/test_processed.csv')
    # Make predictions
    predictions = make_predictions(model, test_data)+1
    
    # Save predictions to a CSV file
    submission_format = pd.read_csv('../data/submission/submission_format.csv', index_col='building_id')
    my_submission = pd.DataFrame(data=predictions,
                             columns=submission_format.columns,
                             index=submission_format.index)
    my_submission.head()   
    my_submission.to_csv('../data/submission/submission.csv')                      
   
