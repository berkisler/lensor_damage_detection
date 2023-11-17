import os
import datetime as dt
import json
import numpy as np


def create_results_directory(base_dir="results", training_dir="training"):
    """
        Create a directory structure for saving training results.

        This function creates a unique directory for each training session under the specified base directory.
        It also creates subdirectories for logs, model weights, and training/test results.

        Parameters:
            base_dir (str): The base directory where the training directories will be created. Default is 'results'.
            training_dir (str): The name of the main training directory under the base directory. Default is 'training'.

        Returns:
            str: The path to the created directory for the current training session.
        """
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
    """
        Update the performance tracker with metrics from the current training session.

        This function updates a JSON file that tracks the performance of different models. It adds or updates
        the entry for the current model identified by its results directory.

        Parameters:
            results_dir (str): The directory of the current training session, used as an identifier for the model.
            metrics (dict): A dictionary containing performance metrics (e.g., loss, accuracy) of the model.
            tracker_file (str): The name of the JSON file used to track model performances. Default is
            'model_performance_tracker.json'.
        """

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
    """
        Compare models based on their performance metrics.

        This function reads the performance tracker file and identifies the best model based on a specific metric
        (e.g., f1-score). It assumes the tracker file contains performance data for multiple models.

        Parameters:
            tracker_file (str): The name of the JSON file used to track model performances. Default is 'model_performance_tracker.json'.

        Returns:
            str: Identifier of the best model based on the chosen metric. Returns None if the tracker file does not exist.
        """
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
