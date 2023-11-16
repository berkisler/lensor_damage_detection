from utils.Generalutil import compare_models
import os

import torch
from torchvision.transforms import functional as F
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def load_model(weights_path, num_classes, device):
    best_model = compare_models()
    best_model_path = os.path.join(best_model, 'model_weights/best_model.pth')

    # Load a pre-trained model for detection and replace the classifier
    model = fasterrcnn_resnet50_fpn(weights='COCO_V1', num_classes=num_classes + 1)  # +1 for background
    model.load_state_dict(torch.load(os.path.join(weights_path, best_model_path), map_location=device))
    model.eval()
    return model


def infer(model, image, device=torch.device('cpu')):
    model.to(device)
    image = F.to_tensor(image).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension
    with torch.no_grad():
        prediction = model(image)
    return prediction[0]  # Assuming single image batch


def run_test_inference(test_dataset, model, device):
    model.to(device)
    test_results = []
    for image, _ in test_dataset:
        prediction = infer(model, image, device)
        test_results.append(prediction)
    return test_results
