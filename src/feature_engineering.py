import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib

train_data = pd.read_csv("./data/processed/train_processed.csv")
test_data = pd.read_csv("./data/processed/test_processed.csv")

X_train = train_data.drop("Potability", axis=1)
y_train = train_data["Potability"]

X_test = test_data.drop("Potability", axis=1)
y_test = test_data["Potability"]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

train_featured = X_train_scaled_df.copy()
train_featured["Potability"] = y_train.values

test_featured = X_test_scaled_df.copy()
test_featured["Potability"] = y_test.values

data_path = os.path.join("data", "featured")
model_path = os.path.join("artifacts")

os.makedirs(data_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

train_featured.to_csv(os.path.join(data_path, "train_featured.csv"), index=False)
test_featured.to_csv(os.path.join(data_path, "test_featured.csv"), index=False)

joblib.dump(scaler, os.path.join(model_path, "scaler.pkl"))

print("Feature engineered data and scaler saved successfully.")
