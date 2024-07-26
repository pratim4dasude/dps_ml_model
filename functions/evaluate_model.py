from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_val, y_val, X_test, y_test):

    y_val_pred = model.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)

    y_test_pred = model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)

    mse_test = mean_squared_error(y_test, y_test_pred)

    print(f'R² score on validation set: {r2_val}')
    print(f'R² score on test set: {r2_test}')
    print(f'Mean Squared Error on test set: {mse_test}')

    return r2_val, r2_test, mse_test
