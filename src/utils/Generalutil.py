import os
import datetime as dt


def create_results_directory(base_dir="results", training_dir="training"):
    # Ensure the base directory and training directory exist
    training_base_dir = os.path.join(base_dir, training_dir)
    os.makedirs(training_base_dir, exist_ok=True)

    # Create a unique timestamped subdirectory
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    training_session_dir = os.path.join(training_base_dir, f"training_{timestamp}")
    os.makedirs(training_session_dir, exist_ok=True)

    # Create subdirectories within the training session directory
    subfolders = ["logs", "model_weights", "train_results", "test_results"]
    for folder in subfolders:
        os.makedirs(os.path.join(training_session_dir, folder), exist_ok=True)

    return training_session_dir


