import os
import pickle


def load_model(model_path, model_class=None, **kwargs):
    """Load model from the given path. If not found, create a new model instance."""
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
    else:
        if model_class:
            model = model_class(**kwargs)
            print(f"New model created for {model_class.__name__} with parameters {kwargs}")
        else:
            raise ValueError("Model not found and no model_class provided for creation.")
    return model


def save_model(model, model_path):
    """Save the model to the given path."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
