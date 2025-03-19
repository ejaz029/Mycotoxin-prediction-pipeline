import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Vomitoxin Concentration")
    plt.ylabel("Predicted Vomitoxin Concentration")
    plt.title("Actual vs Predicted")
    plt.show()
