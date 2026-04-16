from fastapi import FastAPI
import pickle
import pandas as pd
from data_model import Water

app = FastAPI(
    title="Water Quality Assessment API",
    description="A machine learning API for assessing whether water is potable based on water quality measurements."
)

with open("/workspaces/water-potability-mlops-pipeline/model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/")
def index():
    return "Welcome bestie 👋 This API helps you check whether your water is safe for drinking."

@app.post("/predict")
def model_predict(water: Water):
    sample = pd.DataFrame({
          'ph' : [water.ph],
        'Hardness' : [water.Hardness],
        'Solids' : [water.Solids],
        'Chloramines' : [water.Chloramines],
        'Sulfate' : [water.Sulfate],
        'Conductivity' : [water.Conductivity],
        'Organic_carbon' : [water.Organic_carbon],
        'Trihalomethanes' : [water.Trihalomethanes],
        'Turbidity' : [water.Turbidity]
    })

    prediction = model.predict(sample)[0]

    if prediction == 1:
        result = "Water is potable"
    else:
        result = "Water is not potable"

    return {
        "prediction": int(prediction),
        "result": result
    }