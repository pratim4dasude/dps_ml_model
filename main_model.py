import pandas as pd
from functions.preprocess import preprocess_data, encode_features
from functions.train_model import train_model
from functions.evaluate_model import evaluate_model
from functions.visualisation import visualize_data
from functions.actualvspredict import plot_actual_vs_predicted

df = preprocess_data('data.csv')
print("--- Preprocess Done ---")

df = encode_features(df)
print("--- Encoding Done ---")

visualize_data(df)
print("--- Visualisation Done ---")

best_xgb, X_val, y_val, X_test, y_test = train_model(df)
print("--- Train Model Done ---")

r2_val, r2_test, mse_test = evaluate_model(best_xgb, X_val, y_val, X_test, y_test)
print("--- Evaluation Done ---")

# -- Test Prediction --

input_data = {
    'MONATSZAHL': [1],
    'AUSPRAEGUNG': [0],
    'JAHR': [2021],
    'MONAT': [1]
}
input_df = pd.DataFrame(input_data)
predicted_value = best_xgb.predict(input_df)
print(f'Predicted value for input data: {predicted_value}')


plot_actual_vs_predicted(y_test, best_xgb.predict(X_test))