# data_preprocessing.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the data: apply one-hot encoding, fill NaN values."""
    df_processed = pd.get_dummies(df)
    df_processed.fillna(0, inplace=True)
    return df_processed

def plot_distributions(df):
    """Plot distributions of selected features."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    sns.histplot(df['foundation_type'], kde=False, ax=ax[0, 0])
    sns.histplot(df['area_percentage'], kde=True, ax=ax[0, 1])
    sns.histplot(df['height_percentage'], kde=True, ax=ax[0, 2])
    sns.histplot(df['count_floors_pre_eq'], kde=True, ax=ax[1, 0])
    sns.histplot(df['land_surface_condition'], kde=False, ax=ax[1, 1])
    sns.histplot(df['has_superstructure_cement_mortar_stone'], kde=False, ax=ax[1, 2])
    plt.tight_layout()

    # Save plot to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return img_base64

def save_plots_to_html(img_base64):
    """Save the plot as HTML file with embedded image."""
    html = f"""
    <html>
    <head>
        <title>Data Distributions</title>
    </head>
    <body>
        <img src="data:image/png;base64,{img_base64}" />
    </body>
    </html>
    """
    with open('output/distributions.html', 'w') as f:
        f.write(html)

if __name__ == "__main__":
    # Load and preprocess training data
    train_data = load_data('../data/raw/train_values.csv')
    train_labels = load_data('../data/raw/train_labels.csv')
    test_data = load_data('../data/raw/test_values.csv')
    
     # Generate and save visualizations
    img_base64 = plot_distributions(train_data)
    save_plots_to_html(img_base64)

    # Applying preprocessing
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

     # Saving the processed data
    train_data.to_csv('../data/processed/train_processed.csv', index=False)
    test_data.to_csv('../data/processed/test_processed.csv', index=False)
