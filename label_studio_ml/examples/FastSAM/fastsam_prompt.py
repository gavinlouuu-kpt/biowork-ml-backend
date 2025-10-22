"""
FastSAMPrompt implementation adapted for dedicated FastSAM server
Extracted from the original FastSAM repository to avoid import conflicts
"""

import cv2
import numpy as np
import torch
from PIL import Image


def image_to_np_ndarray(image):
    if type(image) is str:
        return np.array(Image.open(image))
    elif issubclass(type(image), Image.Image):
        return np.array(image)
    elif type(image) is np.ndarray:
        return image
    return None


class FastSAMPrompt:

    def __init__(self, image, results, device='cuda'):
        if isinstance(image, str) or isinstance(image, Image.Image):
            image = image_to_np_ndarray(image)
        self.device = device
        self.results = results
        self.img = image
    
    def _format_results(self, result, filter=0):
        annotations = []
        n = len(result.masks.data)
        for i in range(n):
            annotation = {}
            mask = result.masks.data[i] == 1.0

            if torch.sum(mask) < filter:
                continue
            annotation['id'] = i
            annotation['segmentation'] = mask.cpu().numpy()
            annotation['bbox'] = result.boxes.data[i]
            annotation['score'] = result.boxes.conf[i]
            annotation['area'] = annotation['segmentation'].sum()
            annotations.append(annotation)
        return annotations

    def _get_bbox_from_mask(self, mask):
        mask = mask.astype(np.uint8)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2, y2 = x1 + w, y1 + h
        if len(contours) > 1:
            for b in contours:
                x_t, y_t, w_t, h_t = cv2.boundingRect(b)
                # Merge multiple bounding boxes into one.
                x1 = min(x1, x_t)
                y1 = min(y1, y_t)
                x2 = max(x2, x_t + w_t)
                y2 = max(y2, y_t + h_t)
            h = y2 - y1
            w = x2 - x1
        return [x1, y1, x2, y2]

    def box_prompt(self, bbox=None, bboxes=None):
        if self.results is None:
            return []
        assert bbox or bboxes
        if bboxes is None:
            bboxes = [bbox]
        max_iou_index = []
        masks = self.results[0].masks.data
        target_height = self.img.shape[0]
        target_width = self.img.shape[1]
        h = masks.shape[1]
        w = masks.shape[2]
        for bbox in bboxes:
            assert (bbox[2] != 0 and bbox[3] != 0)
            if h != target_height or w != target_width:
                bbox = [
                    int(bbox[0] * w / target_width),
                    int(bbox[1] * h / target_height),
                    int(bbox[2] * w / target_width),
                    int(bbox[3] * h / target_height),
                ]
            bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
            bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
            bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
            bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

            bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
            masks_area = torch.sum(masks[:, bbox[1]:bbox[3], bbox[0]:bbox[2]], dim=(1, 2))
            orig_masks_area = torch.sum(masks, dim=(1, 2))

            union = bbox_area + orig_masks_area - masks_area
            IoUs = masks_area / union
            max_iou_index.append(int(torch.argmax(IoUs)))

        max_iou_index = list(set(max_iou_index))
        return np.array(masks[max_iou_index].cpu().numpy())

    def point_prompt(self, points, pointlabel):  # numpy
        if self.results is None:
            return []
        masks = self._format_results(self.results[0], 0)
        if not masks:
            return []
        target_height = self.img.shape[0]
        target_width = self.img.shape[1]
        h = masks[0]['segmentation'].shape[0]
        w = masks[0]['segmentation'].shape[1]
        if h != target_height or w != target_width:
            points = [[int(point[0] * w / target_width), int(point[1] * h / target_height)] for point in points]

        # Filter out-of-bounds points strictly; don't clamp
        valid_points = []
        valid_labels = []
        for i, pt in enumerate(points):
            xi = int(pt[0])
            yi = int(pt[1])
            if 0 <= xi < w and 0 <= yi < h:
                valid_points.append([xi, yi])
                valid_labels.append(pointlabel[i])
        if not valid_points:
            return []
        points = valid_points
        pointlabel = valid_labels
        onemask = np.zeros((h, w))
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        for annotation in masks:
            if isinstance(annotation, dict):
                mask = annotation['segmentation']
            else:
                mask = annotation
            for i, point in enumerate(points):
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 1:
                    onemask[mask] = 1
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 0:
                    onemask[mask] = 0
        onemask = onemask >= 1
        return np.array([onemask])

    def everything_prompt(self):
        if self.results is None:
            return []
        return self.results[0].masks.data


