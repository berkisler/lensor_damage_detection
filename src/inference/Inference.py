from utils.Generalutil import compare_models
from utils.Datautil import denormalize
from models.Models import CustomObjectDetector
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.transforms import functional as F


def load_model(weights_path, num_classes, device):
    """
        Load the best performing model based on previous training sessions.

        This function compares different models using the 'compare_models' function, identifies the best model,
        and loads its weights.

        Parameters:
            weights_path (str): The base directory where model weights are stored.
            num_classes (int): The number of classes used in the model.
            device (torch.device): The device (CPU or CUDA) on which the model will be loaded.

        Returns:
            torch.nn.Module: The loaded PyTorch model ready for inference.
        """
    best_model = compare_models()
    best_model_path = os.path.join(best_model, 'model_weights/best_model.pth')

    # Create a detector object and load a pre-trained model for detection
    detector = CustomObjectDetector(num_classes, device=device, infer=True,
                                    weight_path=os.path.join(weights_path, best_model_path))

    return detector.model


def infer(model, image, device=torch.device('cpu')):
    """
        Perform inference on a single image using the specified model.

        Args:
            model (torch.nn.Module): The trained model for inference.
            image (PIL.Image or numpy.ndarray): The image on which inference is to be performed.
            device (torch.device): The device to use for inference. Default is CPU.

        Returns:
            dict: The output from the model, typically including detected objects and their properties.
        """
    model.to(device)
    image = F.to_tensor(image).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension
    with torch.no_grad():
        prediction = model(image)
    return prediction[0]  # Assuming single image batch


def run_test_inference(test_loader, model, device):
    """
        Run inference on a test dataset using the specified model.

        This function iterates over a DataLoader containing test data, performs inference on each batch,
        and collects the predictions.

        Parameters:
            test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
            model (torch.nn.Module): The trained model for inference.
            device (torch.device): The device to use for running the model.

        Returns:
            dict: A dictionary where keys are image IDs and values are tuples of (prediction, target) for each image.
        """
    model.to(device)
    test_results = {}

    # Iterate over the DataLoader
    tqdm_bar = tqdm(test_loader, total=len(test_loader))
    for i, batch in enumerate(tqdm_bar):
        images, targets = batch
        images = [img.to(device) for img in images]  # Move images to the specified device
        targets = [{k: v for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)
            for pred, target in zip(predictions, targets):
                test_results[target['image_id'].item()] = (pred, target)

    return test_results


def visualize_prediction(image_tensor, prediction, ground_truth=None, image_id=None):
    """
    Visualize the image with both predicted and ground truth bounding boxes.

    Args:
        image_tensor (torch.Tensor): The image tensor to visualize.
        prediction (dict): The prediction dictionary containing 'boxes', 'labels', and 'scores'.
        ground_truth (dict, optional): The ground truth dictionary containing 'boxes' and 'labels'.
        image_id (int, optional): The ID of the image, if available.
    """
    # Convert image tensor to PIL Image for visualization
    denorm_img = denormalize(image_tensor)
    image = F.to_pil_image(denorm_img)

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    # Draw predicted bounding boxes
    pred_boxes = prediction['boxes'].cpu().numpy()
    pred_labels = prediction['labels'].cpu().numpy()
    pred_scores = prediction['scores'].cpu().numpy()

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x, y, x2, y2 = box
        rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, f'Pred: {label} {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    # Draw ground truth bounding boxes, if provided
    if ground_truth is not None:
        gt_boxes = ground_truth['boxes'].cpu().numpy()
        gt_labels = ground_truth['labels'].cpu().numpy()

        for box, label in zip(gt_boxes, gt_labels):
            x, y, x2, y2 = box
            rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            plt.text(x, y2, f'GT: {label}', bbox=dict(facecolor='green', alpha=0.5))

    if image_id is not None:
        plt.title(f'Image ID: {image_id}')
    plt.show()


def get_user_input_and_visualize(test_loader, test_results):
    """
    Keep asking for user input for bag_id and visualize the corresponding image and prediction.
    Exits the loop when the user types 'done'.

    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        test_results (list): List of predictions for the test dataset.
    """
    max_id = len(test_loader.dataset) - 1

    while True:
        user_input = input(f"Enter a bag_id (image_id) between 0 and {max_id}, or type 'done' to exit: ")

        if user_input.lower() == 'done':
            break

        try:
            bag_id = int(user_input)
            if 0 <= bag_id <= max_id:
                # Retrieve the corresponding image and prediction
                image_tensor, target = test_loader.dataset[bag_id]
                prediction = test_results[bag_id][0]
                gt_box = test_results[bag_id][1]
                image_id = target['image_id'].item() if 'image_id' in target else None
                visualize_prediction(image_tensor, prediction, gt_box,  image_id=image_id)
            else:
                print("Invalid bag_id. Please enter a valid bag_id.")
        except ValueError:
            print("Invalid input. Please enter a numerical bag_id or 'done'.")
