#!/bin/bash

# Exit script on error
set -e

# the directory where your data is stored
DATA_DIR="../data"

# the directory where the model and submissions will be saved
MODEL_DIR="../models"
SUBMISSION_DIR="../data/submission"

echo "Starting the data preprocessing..."
python data_processing.py

echo "Data preprocessing completed. Training the model..."
python train_model.py

echo "Model training completed. Generating predictions..."
python predict_model.py

echo "Predictions generated and saved to $SUBMISSION_DIR/submission.csv"
echo "Pipeline completed successfully."
