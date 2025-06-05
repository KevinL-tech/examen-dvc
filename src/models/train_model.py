import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").iloc([:, 0])

with open("models/best_params.pkl", "rb") as file :
    best_params = pickle.load(file)

model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

with open("models/train_model.pkl", "wb") as f :
    pickle.dump(model, f)


