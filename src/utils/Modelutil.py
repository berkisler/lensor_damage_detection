from collections import defaultdict
import logging
from torchvision.ops import box_iou
import torch


def setup_logger():
    """
    Sets up a logger for logging information.

    This function configures a logger with two handlers:
    - A console handler that outputs log messages to the standard output.
    - A file handler that writes log messages to a file at '../../results/logs/example.log'.

    Both handlers use the INFO level, meaning that all messages at this level and above (WARNING, ERROR, CRITICAL)
    will be processed. The format for the log messages includes the timestamp, log level, and the actual log message.

    Usage:
        Call this function at the beginning of a script to initialize logging.
        After calling, use logging.info(), logging.warning(), etc., to log messages.

    Example:
        setup_logger()
        logging.info("This is an info message")
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Add formatter to handler
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Create file handler and set level to info
    file_handler = logging.FileHandler('../../results/logs/example.log')
    file_handler.setLevel(logging.INFO)

    # Add formatter to file handler
    file_handler.setFormatter(formatter)

    # Add file handler to logger
    logger.addHandler(file_handler)


def tensor_to_list(obj, ids):
    """
        Recursively converts PyTorch tensors in a nested structure to lists.

        This function is useful for converting complex data structures containing PyTorch tensors
        into a format that is serializable, for example, to JSON.

        Args:
            obj (torch.Tensor, dict, list, or any serializable type):
                The object to convert. This can be a PyTorch tensor, a dictionary, a list,
                or any other type. If it's a dictionary or a list, the conversion is applied
                recursively to its elements.

        Returns:
            The converted object where all PyTorch tensors have been replaced with lists.
            If the input is a dictionary or a list, the structure is preserved.

        Usage:
            Useful for preparing data for serialization, such as converting model outputs
            before saving them as JSON.

        Example:
            result = tensor_to_list(torch.tensor([1, 2, 3]))  # returns [1, 2, 3]
            result = tensor_to_list({'tensor': torch.tensor([1, 2, 3]), 'number': 5})
            # returns {'tensor': [1, 2, 3], 'number': 5}
        """

    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()  # Move tensor to CPU and convert to list
    elif isinstance(obj, dict):
        obj['image_id'] = ids
        return {k: tensor_to_list(v, ids) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v, ids) for v in obj]
    else:
        return obj


def calculate_metrics(targets, outputs, iou_threshold=0.4):
    """
    Calculates object detection metrics such as precision, recall, and F1 score.

    Args:
        targets (list): Ground truth values.
        outputs (list): Predicted values.
        iou_threshold (float): Threshold for IoU to consider a match.

    Returns:
        dict: Dictionary containing calculated metrics.
    """
    metrics = defaultdict(list)

    # Calculate and log metrics
    for target, output in zip(targets, outputs):
        if 'boxes' in target and 'boxes' in output:
            # Extract ground truth and predictions
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            pred_boxes = output['boxes']
            pred_labels = output['labels']
            pred_scores = output['scores']

            tp = 0  # True Positives
            fp = 0  # False Positives
            fn = 0  # False Negatives
            tn = 0  # True Negatives

            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                # Calculate IoU for ground truth and predictions
                iou_matrix = box_iou(pred_boxes, gt_boxes)
                # Match predictions and ground truth
                gt_matched = set()
                pred_matched = set()
                for pred_idx, gt_idx in enumerate(torch.argmax(iou_matrix, dim=1)):
                    if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
                        if gt_labels[gt_idx] == pred_labels[pred_idx] and gt_idx not in gt_matched:
                            gt_matched.add(gt_idx)
                            pred_matched.add(pred_idx)

                # Calculate precision, recall, and F1 score
                tp = len(pred_matched)  # true positives
                fp = len(pred_labels) - tp  # false positives
                fn = len(gt_labels) - tp  # false negatives

            elif len(gt_boxes) == 0 and len(pred_boxes) == 0:
                tn = 1  # Correctly identified no objects
            elif len(gt_boxes) == 0:
                fp = len(pred_boxes)  # Predicted objects where there are none
            elif len(pred_boxes) == 0:
                fn = len(gt_boxes)  # Missed detecting objects

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['true_negatives'].append(tn)

    return metrics


def save_best_model(model, current_score, best_score, filename='best_model.pth', mode='min'):
    """
    Saves the model if the current score is better than the best score.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        current_score (float): The current performance score (e.g., validation loss or accuracy).
        best_score (float): The best performance score achieved so far.
        filename (str): Path to the file where the model should be saved.
        mode (str): 'min' if the goal is to minimize the score (e.g., loss),
                    'max' if the goal is to maximize the score (e.g., accuracy).

    Returns:
        float: Updated best score.
    """
    if mode == 'min' and current_score < best_score:
        print(f"New best score achieved: {current_score}. Saving model...")
        torch.save(model.state_dict(), filename)
        best_score = current_score
    elif mode == 'max' and current_score > best_score:
        print(f"New best score achieved: {current_score}. Saving model...")
        torch.save(model.state_dict(), filename)
        best_score = current_score

    return best_score
