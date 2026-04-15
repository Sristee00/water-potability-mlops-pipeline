import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test_data = pd.read_csv("./data/featured/test_featured.csv")

X_test = test_data.drop("Potability", axis=1)
y_test = test_data["Potability"]

model = pickle.load(open("model.pkl", "rb"))

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

metrics_dict = {
    "accuracy": acc,
    "precision": pre,
    "recall": recall,
    "f1_score": f1score
}

with open("metrics.json", "w") as file:
    json.dump(metrics_dict, file, indent=4)

print(metrics_dict)
print("Model evaluation completed successfully.")