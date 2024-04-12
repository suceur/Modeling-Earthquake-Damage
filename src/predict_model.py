# predict_model.py
import pandas as pd
import joblib

def load_model(path):
    return joblib.load(path)

def make_predictions(model, X):
    return model.predict(X)

if __name__ == "__main__":
    model = load_model('../models/best_model.pkl')
    test_data = pd.read_csv('../data/processed/test_processed.csv')
    predictions = make_predictions(model, test_data)+1
    
    train_labels1 = pd.read_csv('../data/raw/train_labels.csv')
    submission_format = pd.read_csv('../data/submission/submission_format.csv', index_col='building_id')
    my_submission = pd.DataFrame(data=predictions,
                             columns=submission_format.columns,
                             index=submission_format.index)
    my_submission.head()   
    my_submission.to_csv('../data/submission/submission.csv')                      
    '''
    submission_df = pd.DataFrame({'building_id': submission_format.columns, 'damage_grade': predictions})
    submission_df.to_csv('../data/submission/submission.csv', index=False)
    submission_df.head()
    '''
