
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml 

n_estimators = yaml.safe_load(open("params.yaml", "r"))["model_building"]["n_estimators"]

train_data = pd.read_csv("./data/featured/train_featured.csv")

X_train = train_data.drop("Potability", axis=1)
y_train = train_data["Potability"]

model = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

pickle.dump(clf, open("model.pkl", "wb"))

print("Model trained and saved successfully.")
