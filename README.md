# Vehicle Damage Detection Challenge

## Introduction
Welcome to the Vehicle Damage Detection Challenge! This challenge focuses on developing AI solutions using deep neural 
networks for object detection to identify damages on vehicles. The repository presents a well-structured pipeline 
that is capable of running fine-tuning steps over a various pretrained models form ***visiontorch***. Furthermore, one
can perform experimentation with various pretrained model options and training hyperparameters and also keep track the
performance on them. Finally perform inference on a test set using the best performing model among the experimented ones
and perform a qualitative check on the predictions.


## The Data
The dataset contains images and annotations in COCO format. The images are RGB and has size 640x640. The annotations 
include both bounding boxes and segmentation polygons. You have the flexibility to choose between tackling the detection 
task using bounding boxes or segmentation polygons. However, currently the modeling pipeline works only for detection task
The dataset comprises 8 classes, with the 'severity-damage' class (id: 0) being a superclass that does not require prediction.

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

**Note**: After cloning the repo. First add the dataset folder to the base path of the repo. Then extract the results.zip 
and place the files under the ``` /results/training ```.

### `main.py` Documentation

#### Overview
The `main.py` script serves as the central entry point for executing the training and inference processes of the object detection task in the Vehicle Damage Detection project. It coordinates different modules to handle dataset loading, model training, evaluation, and inference.

#### Functionality

##### Main Function
- **Signature**: `main(dataset_check=True)`
- **Purpose**: Orchestrates the overall process of training and inference.
- **Parameters**:
  - `dataset_check` (bool): If set to `True`, performs additional dataset analysis and visualization. Default is `True`.

##### Process Flow
1. **Hyperparameters Setup**: Initializes learning rate, momentum, decay rate, scheduler step size, and scheduler gamma for the training process.
2. **Dataset Loading**:
   - Loads training, validation, and test datasets using custom data loaders.
   - Paths for images and annotations are specified for each dataset.
3. **Dataset Analysis and Visualization** (Optional):
   - If `dataset_check` is `True`, the script analyzes the dataset for bounding box statistics and visualizes a sample.
4. **Model Training**:
   - Initializes the `CustomObjectDetector` with the specified number of classes.
   - Sets up optimizer and learning rate scheduler.
   - Trains the model if `training` is `True`, and saves training metrics.
5. **Model Evaluation**:
   - Evaluates the model on the test dataset and saves test metrics.
6. **Model Performance Tracking**:
   - Updates performance metrics for each model in a tracking file.
7. **Inference**:
   - If `inference` is `True`, loads the best performing model based on previous training sessions.
   - Runs inference on the test dataset using the loaded model.
   - Visualizes predictions from the test dataset.

##### Dataset Check
- If enabled, the script calculates bounding box statistics and visualizes a sample from the dataset to ensure data quality.

##### Training and Inference Control
- Controlled by `training` and `inference` boolean flags within the script.

##### Model Loading and Inference
- The script uses utility functions like `load_model` and `run_test_inference` from the `inference` module for model loading and running inference.

##### Dependencies
- The script heavily relies on custom modules like `Dataset`, `CustomObjectDetector`, and various utility functions from `Datautil`, `Generalutil`, and `Inference` modules.




## Note on Fine-tuned model performance
The project uses a pre-trained (on COCO 2017 dataset) Fast R-CNN model (see https://pytorch.org/vision/stable/models/faster_rcnn.html), 
which is then fine-tuned to suit the specific needs of vehicle damage detection. For our use-case the objects in the 
images are rather smaller than the common object that can be found in everyday items. Thus, due to the additional 
challenge of detecting small objects, with default settings the model is performing rather poorly on the vehicle damage 
detection dataset.Moreover, I had some struggles to use GPU computing which really limited my ability to run different 
experiments and improve the model performance. Consequently, the best performing model achieved 0.045 f1-score and 
0.3 recall value. However, given more time, I believe with a more tedious experimentation the performance could be 
increased.

Below are some techniques that I tried/thought to try:

- Apply augmentation/transformation to the training images 
  - Observation: Several different transformations are explored. However, it is observed that this leads to overfitting 
    on training data. This behaviour can be caused by the low number of training data and applying transformation on this
    rather small dataset might increase the discrepancy between train and test/val sets rather than making it more robust
    to variations.
- Change the anchor boxes and aspect ratios 
  - Observation: Since we are trying to detect rather small objects. I tried to introduce smaller anchor boxes by also checking
    distributions of height/width/aspect ratio of the ground truth bounding boxes. Although this created a minor
    improvement, the amount of training data is rather small to change a major model architecture (anchor boxes).
- (Thought) - Applying segmentation instead of bounding box
  - The underlying pipeline is mostly compatible to accommodate the segmentation task as well. The metrics for segmentation
    task needs to be updated. I think it would be a promising alternative to tackle the small sized object detection. 
    However, I could not manage to finish that in time.
- (Thought) - Applying the Slicing Aided Hyper Inference (SAHI) during inference
  - THis technique is based on slicing the input image to smaller overlapping pathces while maintaining the aspect
    ratio. Then running inference on each of them and aggregating the results back together. Focusing the on smaller 
    regions individually first would potentially increase the performance.
