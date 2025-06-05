import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import yaml

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").iloc[:, 0]

model = RandomForestRegressor(random_state=42)

with open("params.yaml", "r") as f :
    params = yaml.safe_load(f)["grid_search"]

param_grid = {
    "n_estimators": params["n_estimators"],
    "max_depth" : params["max_depth"],
    "min_samples_split" : params["min_samples_split"]}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_root_mean_squared_error")
grid_search.fit(X_train, y_train)

with open("models/best_params.pkl", "wb") as file :
    pickle.dump(grid_search.best_params_, file)