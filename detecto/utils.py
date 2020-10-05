import cv2
import os
import pandas as pd
import torch
import xml.etree.ElementTree as ET

from glob import glob
from torchvision import transforms


def default_transforms():

    return transforms.Compose([transforms.ToTensor(), normalize_transform()])


def filter_top_predictions(labels, boxes, scores):

    filtered_labels = []
    filtered_boxes = []
    filtered_scores = []
    # Loop through each unique label
    for label in set(labels):
        # Get first index of label, which is also its highest scoring occurrence
        index = labels.index(label)

        filtered_labels.append(label)
        filtered_boxes.append(boxes[index])
        filtered_scores.append(scores[index])

    if len(filtered_labels) == 0:
        return filtered_labels, torch.empty(0, 4), torch.tensor(filtered_scores)
    return filtered_labels, torch.stack(filtered_boxes), torch.tensor(filtered_scores)


def normalize_transform():

    # Default for PyTorch's pre-trained models
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def read_image(path):

    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def reverse_normalize(image):

    reverse = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
    return reverse(image)


def split_video(video_file, output_folder, prefix='frame', step_size=1):

    # Set step_size to minimum of 1
    if step_size <= 0:
        print('Invalid step_size for split_video; defaulting to 1')
        step_size = 1

    video = cv2.VideoCapture(video_file)

    count = 0
    index = 0
    # Loop through video frame by frame
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Save every step_size frames
        if count % step_size == 0:
            file_name = '{}{}.jpg'.format(prefix, index)
            cv2.imwrite(os.path.join(output_folder, file_name), frame)
            index += 1

        count += 1

    video.release()
    cv2.destroyAllWindows()


def xml_to_csv(xml_folder, output_file=None):

    xml_list = []
    # Loop through every XML file
    for xml_file in glob(xml_folder + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Each object represents each actual image label
        for member in root.findall('object'):
            box = member.find('bndbox')
            label = member.find('name').text

            # Add image file name, image size, label, and box coordinates to CSV file
            row = (filename, width, height, label, int(box[0].text),
                   int(box[1].text), int(box[2].text), int(box[3].text))
            xml_list.append(row)

    # Save as a CSV file
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_names)

    if output_file is not None:
        xml_df.to_csv(output_file, index=None)

    return xml_df


# Checks whether a variable is a list or tuple only
def _is_iterable(variable):
    return isinstance(variable, list) or isinstance(variable, tuple)
