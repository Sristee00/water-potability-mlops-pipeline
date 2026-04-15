import pandas as pd
import numpy as np
import os

train_data = pd.read_csv("./data/raw/train.csv")
test_data = pd.read_csv("./data/raw/test.csv")

median_values = train_data.median()

train_processed_data = train_data.fillna(median_values)
test_processed_data = test_data.fillna(median_values)

data_path = os.path.join("data", "processed")

os.makedirs(data_path, exist_ok=True)

train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

print("Preprocessed train and test data saved successfully.")