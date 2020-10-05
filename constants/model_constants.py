# the classes to train, should match with the class name as labelled
CLASSES = ['car'] 

# training image directory
TRAIN_IMAGES = "output/training_images/"

# training labels path/directory
TRAIN_LABELS = "output/training_labels/"

# validations images path/directory
VALIDATE_IMAGES = "output/validate_images/"

# training labels path/directory
VALIDATE_LABELS = "output/validate_labels/"

# CSV format for training labels
CSV_TRAIN_LABELS = 'train_labels.csv'

# CSV format for validations labels
CSV_VALIDATE_LABELS = 'val_labels.csv'

# Learning Rate
SPLIT_RATIO = 0.1

# Batch Size
BATCH_SIZE = 2

# Learning Rate
LEARNING_RATE = 0.001

# Model Name
MODEL_NAME = 'model_weights_'

# Image to detect
IMAGE_DETECT = 'car_test.jpg'

# Video to detect
INPUT_VIDEO = 'Mercedes.mp4'

# Dectected video
OUTPUT_VIDEO = 'output_vid_Mercedes.avi'
