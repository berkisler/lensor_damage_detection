from dataset import Dataset as ds
import os
import torch
from models.Models import CustomObjectDetector
from utils.Datautil import visualize_sample, analyze_dataset
from utils.Generalutil import create_results_directory
import json
import pickle


def main(dataset_check=True):
    LR = 0.005
    LR_MOMENTUM = 0.9
    LR_DECAY_RATE = 0.0005

    LR_SCHED_STEP_SIZE = 10
    LR_SCHED_GAMMA = 0.1

    base_img_path = 'vehicle_damage_detection_dataset/images/'
    base_ann_path = 'vehicle_damage_detection_dataset/annotations/instances_{}.json'
    task = 'detection'

    tr_img_path = os.path.join(base_img_path, 'train')
    tr_ann_path = base_ann_path.format('train')

    # Create the folder to store the outputs of the training
    result_path = create_results_directory(base_dir="results", training_dir="training")

    train_ds_loader = ds.create_data_loader(tr_img_path, 'imagenet', task, tr_ann_path, 8, train=True)
    print('Train data has been loaded and created: \n')
    # ds.pixel_stats(train_ds_loader)

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
    detector = CustomObjectDetector(num_classes)
    params = [p for p in detector.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=LR_MOMENTUM, weight_decay=LR_DECAY_RATE)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=LR_SCHED_STEP_SIZE,
                                                   gamma=LR_SCHED_GAMMA)

    train_metrics = detector.train_model(train_ds_loader, optimizer, lr_scheduler, num_epochs=1,
                                         val_dataloader=val_ds_loader, )
    # Save train/val dictionary as a JSON file
    with open(os.path.join(result_path, 'train_results/train_metrics.json'), 'w') as tr_json_file:
        json.dump(train_metrics, tr_json_file)

    test_metrics = detector.evaluate_model('test', test_ds_loader)
    # Save test dictionary as a JSON file
    with open(os.path.join(result_path, 'test_results/test_metrics.json'), 'w') as te_json_file:
        json.dump(test_metrics, te_json_file)


if __name__ == '__main__':
    main()
