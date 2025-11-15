import os
import cv2
import numpy as np
import xml.etree.ElementTree as et
from classes import id_classes, img_size


grid_size = (14, 14)
num_anchors = 3
anchors = [[0.05, 0.08], [0.12, 0.15], [0.25, 0.30]]


def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, image_width, image_height):
    x_center = (x_min + x_max) / 2.0 / image_width
    y_center = (y_min + y_max) / 2.0 / image_height

    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    return x_center, y_center, width, height


def get_grid_cell(x_center, y_center, grid_w, grid_h):
    grid_x = int(x_center * grid_w)
    grid_y = int(y_center * grid_h)

    grid_x = max(0, min(grid_x, grid_w - 1))
    grid_y = max(0, min(grid_y, grid_h - 1))

    return grid_x, grid_y


def find_best_anchor(width, height):
    best_iou = best_id = 0

    for i, (anchor_w, anchor_h) in enumerate(anchors):
        intersection = min(width, anchor_w) * min(height, anchor_h)
        union = width * height + anchor_w * anchor_h - intersection

        iou = intersection / union if union > 0 else 0

        if iou > best_iou:
            best_iou = iou
            best_id = i

    return best_id
