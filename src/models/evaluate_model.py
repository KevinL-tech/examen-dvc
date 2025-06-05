import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle, json
import numpy as np

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").iloc[:, 0]

with open("models/train_model.pkl", "rb") as file :
    model = pickle.load(file)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

predictions = pd.DataFrame({
    "y_true" : y_test,
    "y_pred" : y_pred })

predictions.to_csv("data/processed_data/predictions.csv", index = False)

scores = {
    "rmse" : rmse,
    "r2" : r2}

with open("metrics/scores.json", "w") as f :
    json.dump(scores, f, indent=4)