import os
import datetime as dt
import json
import numpy as np


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


def update_performance_tracker(results_dir, metrics, tracker_file="model_performance_tracker.json"):
    tracker_path = os.path.join("results", "training", tracker_file)
    model_identifier = os.path.basename(results_dir)

    # Load existing data from the tracker
    if os.path.isfile(tracker_path):
        with open(tracker_path, 'r') as file:
            tracker_data = json.load(file)
    else:
        tracker_data = {}
    # Update tracker data with the current model's metrics
    tracker_data[model_identifier] = metrics

    # Save updated tracker data
    with open(tracker_path, 'w') as file:
        json.dump(tracker_data, file, indent=4)


def compare_models(tracker_file="model_performance_tracker.json"):
    tracker_path = os.path.join("results", "training", tracker_file)
    if os.path.isfile(tracker_path):
        with open(tracker_path, 'r') as file:
            tracker_data = json.load(file)

        # Process and compare the models
        # Example: Finding the model with the highest accuracy
        best_model = max(tracker_data.items(), key=lambda x: np.mean(x[1].get('f1', 0)))
        print(f"Best model based on f1-score: {best_model[0]} with avg. f1-score {np.mean(best_model[1]['f1'])}")
        return best_model[0]
    else:
        print("No tracker file found.")
        return None
