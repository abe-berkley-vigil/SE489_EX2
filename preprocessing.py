import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_path):
    """
    Load and preprocess data for ML model
    """
    print(f"Loading data from {data_path}")
    # Simulate loading data
    data = pd.DataFrame(np.random.random((100, 5)),
                       columns=["feature_" + str(i) for i in range(5)])

    # Apply standard scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    print("Data preprocessing complete")
    return pd.DataFrame(scaled_data, columns=data.columns)

if __name__ == "__main__":
    processed_data = preprocess_data("data/raw_data.csv")
    print(f"Processed data shape: {processed_data.shape}")