import pandas as pd
df = pd.read_csv('data.csv')

df=df[[
    'MONATSZAHL',   # Category
    'AUSPRAEGUNG',    # Accident-type
    'JAHR',   # Year
    'MONAT',  # Month
    'WERT'   # Value
]]

df_after_2020 = df[df['JAHR'] > 2020]

df = df[df['JAHR'] <= 2020]

# print(df)

# print(df['MONATSZAHL'].value_counts())
# print(df['AUSPRAEGUNG'].value_counts())

df_2000 = df[df['JAHR'] == 2000]

summary_df = df_2000.groupby(['MONATSZAHL', 'AUSPRAEGUNG'])['WERT'].sum().reset_index()

summary_df['MONAT'] = 'Summe'
summary_df['JAHR'] = '2000'

df_updated = pd.concat([df, summary_df], ignore_index=True)

df=df_updated
#-------------------------------------- Ecodeing the data values as label encodeing -----------------------------------------------------------------------

manual_encoding = {
    'Verkehrsunfälle': 0,
    'Alkoholunfälle': 1,
    'Fluchtunfälle': 2
}


df['MONATSZAHL'] = df['MONATSZAHL'].map(manual_encoding)

manual_encoding = {
    'insgesamt': 0,
    'Verletzte und Getötete': 1,
    'mit Personenschäden': 2
}

df['AUSPRAEGUNG'] = df['AUSPRAEGUNG'].map(manual_encoding)

def extract_month(month_str):
    if month_str == "Summe":
        return 0
    else:
        return int(month_str[-2:])

df['MONAT'] = df['MONAT'].apply(extract_month)

df['MONAT'] = df['MONAT'].astype(int)
df['WERT'] = df['WERT'].astype(int)
df['JAHR'] = df['JAHR'].astype(int)

#-------------------------------------- Sort the data with year wise -----------------------------------------------------------------------

# df_sorted = df.sort_values(by=['JAHR', 'MONAT'], ascending=[False, False])
df_sorted = df.sort_values(by=['JAHR', 'MONAT'])
df=df_sorted

df_zero_month = df[df['MONAT'] == 0]
df_filtered = df[df['MONAT'] != 0]

df=df_filtered

# 'MONATSZAHL',  # Category
# 'AUSPRAEGUNG', # Accident-type
# 'JAHR',        # Year
# 'MONAT',       # Month
# 'WERT'         # Value


# split_index = int(len(df) * 0.2)
#
# df_test = df.iloc[:split_index]
# df_train = df.iloc[split_index:]
#
# X_train = df_train.drop(columns='WERT')
# X_test = df_test.drop(columns='WERT')
# y_train = df_train['WERT']
# y_test = df_test['WERT']


# PREDICTION -----------------------------------------------

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


X = df[['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT']]
y = df['WERT']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

param_grid = {
    'n_estimators': [150,200,250],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.04, 0.05, 0.06, 0.07],
    # 'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [2, 3, 4]
}

# XGBRegressor
xgb = XGBRegressor(random_state=42, objective='reg:squarederror')

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')

best_xgb = XGBRegressor(**best_params, random_state=42, objective='reg:squarederror')
best_xgb.fit(X_train, y_train)

y_val_pred = best_xgb.predict(X_val)
r2_val = r2_score(y_val, y_val_pred)
print(f'R² score on validation set: {r2_val}')

y_test_pred = best_xgb.predict(X_test)
r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'R² score on test set: {r2_test}')
print(f'Mean Squared Error on test set: {mse_test}')

y_pred_gb = best_xgb.predict(X_test)

mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print(f'Mean Squared Error on test set for Gradient Boosting: {mse_gb}')
print(f'R² score on test set for Gradient Boosting: {r2_gb}')

# 'MONATSZAHL',  # Category
# 'AUSPRAEGUNG', # Accident-type
# 'JAHR',        # Year
# 'MONAT',       # Month
# 'WERT'         # Value

input_data = {
    'MONATSZAHL': [1],
    'AUSPRAEGUNG': [0],
    'JAHR': [2021],
    'MONAT': [1]
}
input_df = pd.DataFrame(input_data)
predicted_value = best_xgb.predict(input_df)
print(predicted_value)



import matplotlib.pyplot as plt

y_test_pred = best_xgb.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='red', label='Actual Values')
plt.scatter(range(len(y_test_pred)), y_test_pred, color='green', label='Predicted Values')

plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Value')

plt.legend()

plt.show()







