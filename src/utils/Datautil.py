import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
from torchvision.datasets import CocoDetection
import itertools


def convert_coco_format_to_pascal_voc(coco_boxes):
    """
    Convert COCO bounding box format (top_left_x, top_left_y, width, height)
    to Pascal VOC format (x_min, y_min, x_max, y_max).

    Args:
        coco_boxes (Tensor): the bounding boxes in COCO format.

    Returns:
        Tensor: the bounding boxes in Pascal VOC format.
    """
    #coco_boxes = coco_boxes.unsqueeze(0)

    # The input coco_boxes is expected to be a tensor of shape [N, 4] where N is the number of boxes
    if coco_boxes.size(0) != 0:
        x_min = coco_boxes[:, 0]
        y_min = coco_boxes[:, 1]
        x_max = x_min + coco_boxes[:, 2]
        y_max = y_min + coco_boxes[:, 3]
        pascal_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=1)
    else:
        pascal_boxes = torch.zeros((0, 4), dtype=torch.float32)

    return pascal_boxes


def custom_collate_fn(batch):
    """
    Custom collate function to handle batching.

    Parameters:
        batch: List of tuples (image, target) from the dataset.

    Returns:
        tuple: Batch of images and targets.
    """
    images, targets = zip(*batch)

    # Convert images to tensors if they aren't already
    images = [F.to_tensor(img) if not isinstance(img, torch.Tensor) else img for img in images]

    # Convert targets from tuple to list
    targets = list(targets)

    return torch.stack(images), targets


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


def visualize_sample(dataloader):
    """
    Visualize a single sample from a PyTorch DataLoader containing images and their bounding boxes.

    Args:
        dataloader (DataLoader): PyTorch DataLoader with a dataset that returns a tuple of images and targets.
    """
    try:
        while True:
            print("\n Entering data check... type 'done' to stop.")
            idx = input('Provide the sample id: ')
            if idx.lower() == 'done':
                print('Exiting data check.')
                return
            num_samples = len(dataloader.dataset)
            # Get a single batch from the dataloader
            if not idx:
                images, targets = next(iter(dataloader))
            else:
                idx = int(idx)
                if idx < num_samples:
                    # Use modulo to wrap around if idx exceeds the dataset length
                    images, targets = next(itertools.islice(dataloader, idx % num_samples, None))
                else:
                    print(f"Index out of range. Dataset contains {num_samples} samples.")
                    continue

            # Assuming that the dataset returns a PIL Image or a tensor that can be converted, and
            # the targets include 'boxes' and 'labels'
            image = images[0]  # Get the first image from the batch
            target = targets[0]  # Get the first target from the batch
            im_id = target['image_id']
            print(f'Image ID" {im_id}')
            print(f"{type(image) = }\n{type(target) = }\n{target.keys() = }")
            print(f"{type(target['boxes']) = }\n{type(target['labels']) = }")

            # Convert tensor to PIL Image if necessary
            if isinstance(image, torch.Tensor):
                image = F.to_pil_image(image)

            # Create figure and axes
            fig, ax = plt.subplots(1)

            # Display the image
            ax.imshow(image)

            # Get the bounding boxes and labels from the target
            boxes = target['boxes'].cpu().numpy()
            labels = target['labels'].cpu().numpy()

            # Create a Rectangle patch for each bounding box and add it to the plot
            for i, box in enumerate(boxes):
                # Create a Rectangle patch
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                         linewidth=1, edgecolor='r', facecolor='none', label=labels[i])

                # Add the patch to the Axes
                ax.add_patch(rect)

            # Add labels for the first box only to avoid duplicate labels in the legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            plt.show()

    except KeyboardInterrupt:
        print('Loop stopped!')

    except StopIteration:
        print("Reached the end of the dataset.")


def analyze_dataset(dataset_path, annotation_file):
    """
    Analyze the dataset to determine appropriate anchor sizes and aspect ratios.

    Parameters:
        dataset_path (str): Path to the dataset directory.
        annotation_file (str): Path to the annotation file.

    Returns:
        A plot of the distribution of bounding box sizes and aspect ratios.
    """
    dataset = CocoDetection(root=dataset_path, annFile=annotation_file)

    widths = []
    heights = []
    aspect_ratios = []
    areas = []

    for _, targets in dataset:
        for target in targets:
            bbox = target['bbox']
            # COCO format: [xmin, ymin, width, height]
            width = bbox[2]
            height = bbox[3]
            area = width * height

            widths.append(width)
            heights.append(height)
            areas.append(area)
            aspect_ratios.append(width / height if height > 0 else 0)

    # Computing max and min areas
    max_area, min_area = max(areas), min(areas)
    print('max_area: ', max_area, 'min_area: ', min_area)

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.hist(widths, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Widths')

    plt.subplot(1, 3, 2)
    plt.hist(heights, bins=50, color='green', alpha=0.7)
    plt.title('Distribution of Heights')

    plt.subplot(1, 3, 3)
    plt.hist(aspect_ratios, bins=50, color='red', alpha=0.7)
    plt.title('Distribution of Aspect Ratios')

    plt.tight_layout()
    plt.show()
