import pickle
import os

def save_model(model, path="models"):
    """
    Save trained model to disk
    """
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}/model.pkl")

def load_model(path="models/model.pkl"):
    """
    Load model from disk
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {path}")
    return model

def deploy_model(model, endpoint="production"):
    """
    Simulate deploying model to production
    """
    print(f"Deploying model to {endpoint} environment")
    # Deployment code would go here
    print("Model deployed successfully")

if __name__ == "__main__":
    # Simulate a model object
    dummy_model = {"type": "RandomForest", "hyperparams": {"n_estimators": 100}}
    save_model(dummy_model)
    loaded_model = load_model()
    deploy_model(loaded_model)