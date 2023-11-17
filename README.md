# Vehicle Damage Detection Challenge

## Introduction
Welcome to the Vehicle Damage Detection Challenge! This challenge focuses on developing AI solutions using deep neural networks for object detection to identify damages on vehicles. The aim is not necessarily to create a state-of-the-art model but to develop a well-structured pipeline demonstrating your proficiency in deep learning and AI software development.

## The Data
Participants will receive a zip file containing images and annotations in COCO format. The annotations include both bounding boxes and segmentation polygons. You have the flexibility to choose between tackling the detection task using bounding boxes or segmentation polygons. The dataset comprises 8 classes, with the 'severity-damage' class (id: 0) being a superclass that does not require prediction.

## The Goal
Your task is to build a neural network capable of effectively detecting vehicle damages from single RGB images. The model's output should include coordinates (set of points), the damage class, and the confidence level of the prediction. You're free to choose any object detection model, like YOLO or RCNN, and leverage pre-trained weights. Ensure that these weights are accessible for us to execute your model. Bonus points are awarded for result visualization, such as overlaying predictions on test images. While Pytorch is preferred, feel free to use any library of your choice.

## Folder Structure
- **models**: Contains model-related classes and functions.
- **notebooks**: Jupyter notebooks for experiments and analyses.
- **requirements**: 
  - `environments.yml`: Conda environment file.
  - `requirements.txt`: Python requirements file.
- **results/training**: Training results and model performance data.
  - `model_performance_tracker.json`: Tracks model performance across training sessions.
  - `training_YYYYMMDD_HHMMSS`: Specific training session folder.
    - **model_weights**: Contains saved model weights (`best_model.pth`).
    - **test_results**: Test evaluation results (`test_metrics.json`).
    - **train_results**: Training evaluation results (`train_metrics.json`).
- **scripts**: Utility scripts for various tasks.
- **src**: Source code for the project.
  - `main.py`: Main script to run training and inference.
  - **model**: Module for model-related functionalities.
  - **dataset**: Module for dataset handling and processing.
  - **inference**: Module for inference and prediction-related functionalities.
  - **utils**: Utility functions supporting other modules.
- `.gitignore`: Specifies files to be ignored in Git version control.
- `README.md`: This documentation file.

## Environment Setup
To set up the project environment:
1. Install Conda and create an environment using `environments.yml`:
   ```bash
   conda env create -f environments.yml 
   conda activate lensor_detection
   pip install -r requirements.txt
   ```
   
## Usage Instructions
`main.py` is the entry point for running the training and inference processes. It accepts a boolean argument `dataset_check` for performing dataset analysis.

- **Training**: Set the `training` boolean to `True` in `main.py` to start the training process.
- **Inference**: Set the `inference` boolean to `True` in `main.py` to run the inference on the test set.

Note: Extract the results.zip and place the files under the /results/training

## Note on Pretrained Model
The project uses a pre-trained Fast R-CNN model, which is then fine-tuned to suit the specific needs of vehicle damage detection.
