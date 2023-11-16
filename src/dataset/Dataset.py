from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import torch
from utils.Datautil import custom_collate_fn, convert_coco_format_to_pascal_voc


class CustomCocoDataset(CocoDetection):
    """
    Custom dataset class for handling COCO format for both detection and segmentation.
    """

    def __init__(self, root, ann_file, task='detection', transform=None):
        """
        Initialize the dataset.

        Parameters:
            root (str): Directory where images are located.
            ann_file (str): Path to COCO annotation file.
            task (str): Task type - 'detection' or 'segmentation'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(CustomCocoDataset, self).__init__(root, ann_file)
        self.task = task
        self.transform = transform

    def __len__(self):
        """
        Return the number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.ids)  # `ids` is a list of annotation IDs, part of CocoDetection

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Parameters:
            idx (int): Index of the item.

        Returns:
            tuple: (image, target) where target depends on the task.
        """
        img, annotations = super(CustomCocoDataset, self).__getitem__(idx)
        # Initialize containers for bbox, labels, and (if needed) masks
        boxes = []
        labels = []
        masks = []  # Only used if task is 'segmentation'
        image_id = annotations[0]['image_id'] if annotations else -1

        for ann in annotations:
            # Extract label
            labels.append(ann['category_id'])
            # Extract bounding box
            boxes.append(ann['bbox'])

            if self.task == 'segmentation':
                # Extract segmentation mask
                masks.append(ann['segmentation'][0])

        converted_box = convert_coco_format_to_pascal_voc(torch.as_tensor(boxes, dtype=torch.float32))
        if self.transform:
            img, converted_box = self.transform(img, converted_box)

        new_target = {
            'boxes': converted_box,
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([image_id])  # Encapsulate image_id in a tensor
        }

        if self.task == 'segmentation':
            new_target['masks'] = torch.as_tensor(masks, dtype=torch.float32)

        return img, new_target

    @staticmethod
    def get_transform(base_data, train=True):
        """
        Returns a composed transform function.

        Parameters:
            base_data (str): The dataset that pretrained model is originally trained with.
            train (bool): Flag to indicate if the transform is for training or validation.

        Returns:
            A transform function composed of several individual transforms.
        """
        pretrained_pixel_stats = {'imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
                                  }
        norm_mean = pretrained_pixel_stats[base_data]['mean']
        norm_std = pretrained_pixel_stats[base_data]['std']

        if train:
            # Example of adding more transforms for training
            return v2.Compose([
                v2.Resize(1024),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(15),
                v2.ToTensor(),
                v2.Normalize(mean=norm_mean, std=norm_std)
            ])
        else:
            # Transforms for validation/testing
            return v2.Compose([
                v2.Resize(1024),
                v2.ToTensor(),
                v2.Normalize(mean=norm_mean, std=norm_std)
            ])


def create_data_loader(dataset_path, base_data, task, annotation_file, batch_size, train=True):
    """
    Creates a data loader for the CustomCocoDataset.

    Parameters:
        dataset_path (str): Path to the dataset directory.
        base_data (str): The dataset that pretrained model is originally trained with.
        task (str): Tge cv task from which the ground truths are extracted
        annotation_file (str): Path to the annotation file.
        batch_size (int): Batch size for the data loader.
        train (bool): Flag indicating training or validation loader.

    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    dataset = CustomCocoDataset(root=dataset_path,
                                ann_file=annotation_file,
                                task=task,
                                transform=CustomCocoDataset.get_transform(base_data, train))

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=train,
                        num_workers=4,
                        collate_fn=custom_collate_fn)

    return loader
