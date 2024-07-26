import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from joblib import dump

# ----------------- Randomforest model ----------------

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# X = df[['Category', 'Accident-type', 'Year', 'Month']]
# y = df['Value']
#
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42,shuffle = False, stratify = None)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42,shuffle = False, stratify = None)
#
# param_grid = {
#     'n_estimators': [20,30,50,60],
#     'max_depth': [5, 10, 15],
#     'min_samples_split': [7,8,10,11,12]
# }
#
# rf = RandomForestRegressor(random_state=42)
#
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
#
# grid_search.fit(X_train, y_train)
#
# best_params = grid_search.best_params_
# print(f'Best parameters: {best_params}')
# best_rf = RandomForestRegressor(**best_params, random_state=42)
# best_rf.fit(X_train, y_train)
#
# y_val_pred = best_rf.predict(X_val)
# r2_val = r2_score(y_val, y_val_pred)
# print(f'R² score on validation set: {r2_val}')
#
# y_test_pred = best_rf.predict(X_test)
# r2_test = r2_score(y_test, y_test_pred)
# mse_test = mean_squared_error(y_test, y_test_pred)
# print(f'R² score on test set: {r2_test}')
# print(f'Mean Squared Error on test set: {mse_test}')

# -----------------------------------

def train_model(df):
    X = df[['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT']]
    y = df['WERT']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    #  Hyper parameter tuning
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.04, 0.05, 0.06, 0.07],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [2, 3, 4]
    }

    xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_xgb = XGBRegressor(**best_params, random_state=42, objective='reg:squarederror')
    best_xgb.fit(X_train, y_train)

    # ---- Storing the model.pkl -----
    dump(best_xgb, 'model.pkl')

    return best_xgb, X_val, y_val, X_test, y_test
