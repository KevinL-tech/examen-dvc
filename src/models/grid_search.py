import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").iloc[:, 0]

model = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth" : [2, 5, 10],
    "min_samples_split" : [1, 3, 5]}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_root_mean_squared_error")
grid_search.fit(X_train, y_train)

with open("models/best_params.pkl", "wb") as file :
    pickle.dump(grid_search.best_params_, file)