# train.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    # 1. Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Train a simple model
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    # 3. Save the model to disk
    joblib.dump(clf, "model.pkl")
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_model()