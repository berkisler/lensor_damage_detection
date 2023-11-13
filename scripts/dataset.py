from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from PIL import Image
import torch


def create_data_loader(dataset_path, annotation_file, batch_size, train=True):
    """
    Creates a data loader for the CustomCocoDataset.

    Parameters:
        dataset_path (str): Path to the dataset directory.
        annotation_file (str): Path to the annotation file.
        batch_size (int): Batch size for the data loader.
        train (bool): Flag indicating training or validation loader.

    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    dataset = CustomCocoDataset(root=dataset_path,
                                annFile=annotation_file,
                                task='detection',  # Change to 'segmentation' if needed
                                transform=CustomCocoDataset.get_transform(train))

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=train,
                        num_workers=4,
                        collate_fn=lambda x: tuple(zip(*x)))

    return loader


class CustomCocoDataset(CocoDetection):
    """
    Custom dataset class for handling COCO format for both detection and segmentation.
    """

    def __init__(self, root, annFile, task='detection', transform=None):
        """
        Initialize the dataset.

        Parameters:
            root (str): Directory where images are located.
            annFile (str): Path to COCO annotation file.
            task (str): Task type - 'detection' or 'segmentation'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(CustomCocoDataset, self).__init__(root, annFile)
        self.task = task
        self.transform = transform

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Parameters:
            idx (int): Index of the item.

        Returns:
            tuple: (image, target) where target depends on the task.
        """
        img, target = super(CustomCocoDataset, self).__getitem__(idx)
        img = Image.fromarray(img).convert("RGB")

        if self.transform:
            img = self.transform(img)

        new_target = {'image_id': target['image_id'], 'area': target['area']}

        if self.task == 'detection':
            # Extract bounding box information
            new_target['truth'] = torch.as_tensor([ann['bbox'] for ann in target['annotations']],
                                                  dtype=torch.float32)

        elif self.task == 'segmentation':
            # Extract segmentation mask information
            new_target['truth'] = torch.as_tensor([ann['segmentation'][0] for ann in target['annotations']],
                                                  dtype=torch.float32)

        new_target['labels'] = torch.as_tensor([ann['category_id'] for ann in target['annotations']],
                                               dtype=torch.int64)
        return img, new_target

    def get_transform(self, train=True):
        """
        Returns a composed transform function.

        Parameters:
            train (bool): Flag to indicate if the transform is for training or validation.

        Returns:
            A transform function composed of several individual transforms.
        """
        transforms_list = [transforms.ToTensor()]
        if train:
            # Example of adding more transforms for training
            transforms_list.extend([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # Add more transformations as needed
            ])
        else:
            # Transforms for validation/testing
            # Example: Resize, Normalize, etc.
            pass

        return transforms.Compose(transforms_list)
