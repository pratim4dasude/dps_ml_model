import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_test, y_test_pred):

    # plt.figure(figsize=(5, 2))

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='red', label='Actual Values')
    plt.scatter(range(len(y_test_pred)), y_test_pred, color='green', label='Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()