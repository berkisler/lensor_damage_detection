from dataset import dataset as ds
import os
import torch
import cv2
from torchvision.transforms import functional as F


def main():
    base_img_path = 'vehicle_damage_detection_dataset/images/'
    base_ann_path = 'vehicle_damage_detection_dataset/annotations/instances_{}.json'
    task = 'detection'

    tr_img_path = os.path.join(base_img_path, 'train')
    tr_ann_path = base_ann_path.format('train')

    train_ds_loader = ds.create_data_loader(tr_img_path, 'imagenet', task, tr_ann_path, 32, train=True)
    print('Train data has been loaded and created: ')
    # ds.pixel_stats(train_ds_loader)

    te_img_path = os.path.join(base_img_path, 'test')
    te_ann_path = base_ann_path.format('test')

    test_ds_loader = ds.create_data_loader(te_img_path, 'imagenet', task, te_ann_path, 32, train=True)
    print('Test data has been loaded and created: ')
    # ds.pixel_stats(test_ds_loader)


if __name__ == '__main__':
    main()