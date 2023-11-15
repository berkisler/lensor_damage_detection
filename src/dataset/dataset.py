from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import torch


def custom_collate_fn(batch):
    """
    Custom collate function to handle batching.

    Parameters:
        batch: List of tuples (image, target) from the dataset.

    Returns:
        tuple: Batch of images and targets.
    """
    images, targets = zip(*batch)
    return images, targets


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


def pixel_stats(loader):
    """
    Calculate the mean and standard deviation of images in a dataset.

    This function iterates over a DataLoader and calculates the mean and
    standard deviation of all images in the dataset. This is useful for
    normalizing the dataset in future data preprocessing steps.

    Parameters:
    - loader (DataLoader): A PyTorch DataLoader object that loads the dataset
      for which the statistics are to be calculated. The DataLoader should
      return batches of images and their corresponding labels or targets.

    Returns:
    - mean (Tensor): A tensor containing the mean value of each channel in the dataset.
    - std (Tensor): A tensor containing the standard deviation of each channel in the dataset.

    Note:
    - This function assumes that the images returned by the DataLoader are
      PyTorch tensors with the channel order being (C, H, W) where C is the
      number of channels, H is the height, and W is the width of the image.
    - The function stacks the images in each batch to create a single tensor
      per batch. This requires that all images in the batch have the same shape.
    - The DataLoader should shuffle the data to ensure a representative
      calculation of the mean and standard deviation if the dataset is too
      large to process in one pass.
    - If the dataset is normalized (e.g., pixel values range between 0 and 1),
      the calculated mean and standard deviation will be in the same range.
    """

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, label in loader:
        images = torch.stack(data)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    # print(f"Calculated mean: {mean}")
    # print(f"Calculated std: {std}")

    return mean, std

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
        # print(annotations)
        if self.transform:
            img = self.transform(img)

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

        new_target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
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
            return transforms.Compose([
                transforms.Resize(640),
                # transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # transforms.RandomRotation(15),
                transforms.ToTensor()
                # transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
        else:
            # Transforms for validation/testing
            return transforms.Compose([
                transforms.Resize(640),
                transforms.ToTensor()
                # transforms.Normalize(mean=norm_mean, std=norm_std)
            ])