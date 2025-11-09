import os
import cv2
import numpy as np
import xml.etree.ElementTree as et
from classes import id_classes, img_size


grid_size = (14, 14)
num_anchors = 3

anchors = [[0.05, 0.08], [0.12, 0.15], [0.25, 0.30]]


def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height):
    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height

    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    return x_center, y_center, width, height
