import torch
import torch.nn as nn
from models.yolov5 import YOLOv5  # Assuming YOLOv5 model is defined in this path

class CustomYOLOv5(nn.Module):
    """
    Customized YOLOv5 model for Vehicle Damage Detection.
    """

    def __init__(self, num_classes, model_path=None):
        """
        Initialize the model.

        Parameters:
            num_classes (int): Number of classes (damage types in this case).
            model_path (str, optional): Path to the pre-trained model weights.
        """
        super(CustomYOLOv5, self).__init__()
        self.model = YOLOv5(num_classes=num_classes)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            output (Tensor): Model output.
        """
        return self.model(x)

