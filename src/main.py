from dataset import Dataset as ds
import os
import torch
from models.Models import CustomObjectDetector
from utils.Datautil import visualize_sample, analyze_dataset
from utils.Generalutil import create_results_directory, update_performance_tracker
import json
import pickle
from inference.Inference import load_model, run_test_inference, get_user_input_and_visualize


def main(dataset_check=True):
    """
        The main function to execute training and inference processes for an object detection task.

        This function initializes data loaders, trains a model (if specified), and runs inference on a test dataset.
        It also includes options to check the dataset and visualize results.

        Parameters:
            dataset_check (bool): If True, performs dataset analysis and visualization before training. Default is True.

        Process Flow:
        1. Set up hyperparameters for training.
        2. Load training, validation, and test datasets.
        3. Optionally, analyze the dataset and visualize some samples.
        4. If training is enabled, train the model and save the training metrics and results.
        5. If inference is enabled, load the best model and run inference on the test dataset.
        6. Visualize the predictions from the test dataset.
        """
    LR = 0.005
    LR_MOMENTUM = 0.9
    LR_DECAY_RATE = 0.0005

    LR_SCHED_STEP_SIZE = 10
    LR_SCHED_GAMMA = 0.1

    base_img_path = 'vehicle_damage_detection_dataset/images/'
    base_ann_path = 'vehicle_damage_detection_dataset/annotations/instances_{}.json'
    task = 'detection'

    training = True
    inference = True

    tr_img_path = os.path.join(base_img_path, 'train')
    tr_ann_path = base_ann_path.format('train')

    train_ds_loader = ds.create_data_loader(tr_img_path, 'imagenet', task, tr_ann_path, 8, train=True)
    print('Train data has been loaded and created: \n')
    # # ds.pixel_stats(train_ds_loader)

    val_img_path = os.path.join(base_img_path, 'val')
    val_ann_path = base_ann_path.format('val')

    val_ds_loader = ds.create_data_loader(val_img_path, 'imagenet', task, val_ann_path, 8, train=False)
    print('Validation data has been loaded and created: \n')

    te_img_path = os.path.join(base_img_path, 'test')
    te_ann_path = base_ann_path.format('test')

    test_ds_loader = ds.create_data_loader(te_img_path, 'imagenet', task, te_ann_path, 8, train=False)
    print('Test data has been loaded and created: \n')
    # ds.pixel_stats(test_ds_loader)

    if dataset_check:
        # Create some basic statistics to understand of the typical sizes and shapes of the
        print('Calculating bbox statistics...')
        analyze_dataset(tr_img_path, tr_ann_path)

        # Visualize some input data to validate the training data
        visualize_sample(val_ds_loader)

    # Initialize the model, optimizer, and train
    num_classes = 8
    if training:
        # Create the folder to store the outputs of the training
        result_path = create_results_directory(base_dir="results", training_dir="training")

        detector = CustomObjectDetector(num_classes)
        params = [p for p in detector.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=LR, momentum=LR_MOMENTUM, weight_decay=LR_DECAY_RATE)

        # Initialize the learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=LR_SCHED_STEP_SIZE,
                                                       gamma=LR_SCHED_GAMMA)

        train_metrics = detector.train_model(train_ds_loader, optimizer, lr_scheduler, num_epochs=1,
                                             res_path=result_path, val_dataloader=val_ds_loader)
        # # Save train/val dictionary as a JSON file
        with open(os.path.join(result_path, 'train_results/train_metrics.json'), 'w') as tr_json_file:
            json.dump(train_metrics, tr_json_file)

        test_metrics = detector.evaluate_model('test', test_ds_loader)
        # Save test dictionary as a JSON file
        with open(os.path.join(result_path, 'test_results/test_metrics.json'), 'w') as te_json_file:
            json.dump(test_metrics, te_json_file)

        # Updating the model performance tracker with the latest results
        update_performance_tracker(result_path, te_json_file)

    if inference:
        base_path = 'results/training/'
        for entry in os.listdir(base_path):
            model_path = os.path.join(base_path, entry)
            if os.path.isdir(model_path):
                metric_path = os.path.join(model_path, 'test_results/test_metrics.json')
                with open(metric_path, 'r') as file:
                    metric_data = json.load(file)
                update_performance_tracker(model_path, metric_data)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_model('results/training', num_classes=8, device=device)

        test_results = run_test_inference(test_ds_loader, model, device)

        get_user_input_and_visualize(test_ds_loader, test_results)


if __name__ == '__main__':
    main(dataset_check=True)
