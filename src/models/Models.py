import logging
from collections import defaultdict
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from tqdm import tqdm
from utils.Modelutil import calculate_metrics, tensor_to_list, save_best_model
import os
import numpy as np



def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CustomObjectDetector:
    """
    A custom object detector class that encapsulates a PyTorch-based Faster R-CNN model for object detection tasks.

    This class provides functionalities for initializing a Faster R-CNN model with a ResNet50 backbone pre-trained on
    the COCO dataset. It includes methods for training the model, evaluating its performance, and running inference.
    The class allows customization of the anchor generator in the Region Proposal Network (RPN) to adapt to different
    object sizes and aspect ratios in the dataset.

    Attributes:
        model (torch.nn.Module): The Faster R-CNN model instance.
        device (torch.device): The device (CPU or GPU) on which the model will be run.
    """
    def __init__(self, num_classes, device=None, infer=None, weight_path=None):
        """
        Initializes the CustomObjectDetector with a pre-trained Faster R-CNN model.

        Args:
            num_classes (int): The number of classes for object detection, including the background class.
            device (torch.device, optional): The device (CPU or GPU) to run the model on. Defaults to GPU if available.
            infer (bool, optional): Flag indicating if the model is being loaded for inference. Defaults to False.
            weight_path (str, optional): Path to the pre-trained model weights, used for inference. Required if infer=True.

        """
        # Initialize the logger
        setup_logger()

        # Load a pre-trained model for detection and return
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')

        # Replace the classifier with a new one for the desired number of classes
        # (num_classes + 1) as we need to include the background as a class
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                                        num_classes + 1)

        # # Create a custom anchor generator for the RPN
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))  # Adjust as needed
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.model.rpn.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        if infer:
            if weight_path:
                self.model.load_state_dict(torch.load(os.path.join(weight_path), map_location=device))
                self.model.eval()
            else:
                raise(ValueError('Path for the model weight is not defined!'))

        else:
            self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Move model to the specified device
            self.model.to(self.device)

    def forward(self, images, targets=None):
        """
        Forward pass of the model.

        Args:
            images (list[Tensor]): Images to be processed by the model.
            targets (list[dict[Tensor]]): Annotations for the images.

        Returns:
            result (list[dict[Tensor]] or dict[Tensor]): The model output.
        """
        return self.model(images, targets)

    def evaluate_model(self, phase, dataloader):
        """
        Evaluates the model on a separate dataset.
        Logs additional metrics such as precision, recall, and F1 score.

        Args:
            phase (str): Indicating whether the evaluation is for validation or test data
            dataloader (DataLoader): Dataloader for the validation or test data.
        """
        # self.model.eval()  # Set the model to evaluation mode
        eval_loss = 0.0
        metrics = defaultdict(list)
        all_predictions = []
        all_losses = []
        logging.info(f'<<<<<<<<<<{phase} started>>>>>>>>>>')
        tqdm_bar = tqdm(dataloader, total=len(dataloader))

        for i, data in enumerate(tqdm_bar):
            with torch.no_grad():  # No need to track gradients for validation
                images, targets = data
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.model.train()
                # Log loss
                loss_dict = self.model(images, targets)

                self.model.eval()
                # Forward pass
                outputs = self.model(images)
                image_ids = [t['image_id'] for t in targets]
                serializable_outputs = [tensor_to_list(output, ids) for ids, output in zip(image_ids, outputs)]

                all_predictions.extend(serializable_outputs)

                losses = torch.sum(torch.stack(list(loss_dict.values())))
                all_losses.append(losses.item())
                metric = calculate_metrics(targets, outputs)

                # Flatten the results of the images
                for k, v in metric.items():
                    metrics[k].append(v)

                means = {metric_var: [np.mean(sublist) for sublist in vals] for metric_var, vals in metrics.items()}
                tqdm_bar.set_description(
                    desc=f"Val -- Loss: {losses.item():.4f} - "
                         f"prec: {np.mean(means['precision']):.4f} - "
                         f"recall: {np.mean(means['recall']):.4f} - "
                         f"f1: {np.mean(means['f1']):.4f} - "
                )

        # Flatten the results of the images within an epoch
        eval_metrics = {k: [item for sublist in v for item in sublist] for k, v in metrics.items()}

        # Calculate average values for metrics and loss
        # avg_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
        eval_metrics[f'{phase}_loss'] = all_losses

        # Log metrics
        logging.info(f'{phase} Avg. Loss: {np.mean(all_losses)}')

        eval_metrics[f'{phase}_predictions'] = all_predictions

        return eval_metrics

    def train_model(self, dataloader, optimizer, lr_scheduler, num_epochs, res_path, val_dataloader=None):

        """
        Trains the model, with an option to validate.

        Args:
            dataloader (DataLoader): Dataloader for the training data.
            optimizer (Optimizer): Optimizer for the model.
            lr_scheduler (lr_scheduler): Learning rate scheduler for the model
            num_epochs (int): Number of epochs to train.
            res_path (str): The path to save the model weights
            val_dataloader (DataLoader, optional): Dataloader for the validation data.
        """
        logging.info('<<<<<<<<<<Training started>>>>>>>>>>')
        all_metrics = defaultdict(dict)
        best_val_loss = float('inf')  # Initialize with a large number for loss minimization

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs} started!')
            self.model.train()  # Set the model to training mode
            total_loss = 0.0
            metrics = defaultdict(list)
            running_loss = 0.0

            tqdm_bar = tqdm(dataloader, total=len(dataloader))
            for i, data in enumerate(tqdm_bar):
                images, targets = data
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = torch.sum(torch.stack(list(loss_dict.values())))
                total_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # Get model predictions
                self.model.eval()  # Set the model to evaluation mode for prediction
                with torch.no_grad():
                    predictions = self.model(images)
                self.model.train()  # Set back to train mode

                metric = calculate_metrics(targets, predictions)

                for k, v in metric.items():
                    metrics[k].append(v)

                means = {metric_var: [np.mean(sublist) for sublist in vals] for metric_var, vals in metrics.items()}
                tqdm_bar.set_description(
                    desc=f"Train -- Loss: {losses.item():.4f} - "
                         f"prec: {np.mean(means['precision']):.4f} - "
                         f"recall: {np.mean(means['recall']):.4f} - "
                         f"f1: {np.mean(means['f1']):.4f} - "
                )

            # Flatten the results of the images within an epoch
            train_metrics = {k: [item for sublist in v for item in sublist] for k, v in metrics.items()}

            # Calculate average values for metrics and loss
            total_loss /= len(dataloader.dataset)
            train_metrics['training_loss'] = total_loss

            # Log metrics
            logging.info(f'Epoch {epoch}/{num_epochs} - Avg. Loss: {total_loss}')
            all_metrics[f'epoch_{epoch}']['train'] = train_metrics

            # Iterate over lr scheduler
            lr_scheduler.step()

            # Perform validation if a validation dataloader is provided
            if val_dataloader:
                eval_metrics = self.evaluate_model('val', val_dataloader)

                all_metrics[f'epoch_{epoch}']['val'] = eval_metrics

            # Calculate the mean loss on eval set to track best performing model
            val_loss = np.mean(eval_metrics['val_loss'])

            # Save model if it's the best so far
            file_path = os.path.join(res_path, 'model_weights/best_model.pth')
            best_val_loss = save_best_model(self.model, val_loss, best_val_loss, filename=file_path, mode='min')

        return all_metrics
