import os
import cv2
import json
import logging
import time
import numpy as np
from skimage.draw import polygon as skimage_polygon

from label_studio_sdk.converter import brush
from typing import List, Dict, Optional
from uuid import uuid4
from .sam_predictor import SAMPredictor, get_credentials_for_task
from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)

# Dedicated FastSAM configuration
RESPONSE_TYPE = os.environ.get("RESPONSE_TYPE", "both")
POLYGON_DETAIL_LEVEL = float(os.environ.get("POLYGON_DETAIL_LEVEL", "0.002"))
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "10"))


class SamMLBackend(LabelStudioMLBase):

    def setup(self):
        self.set("model_version", f"{self.__class__.__name__}-fastsam-v1")

    def get_runtime_config(self, **kwargs):
        response_type = kwargs.get('response_type') or RESPONSE_TYPE
        polygon_detail_level = float(kwargs.get('polygon_detail_level') or POLYGON_DETAIL_LEVEL)
        max_results = int(kwargs.get('max_results') or MAX_RESULTS)
        if response_type not in ['brush', 'polygon', 'both']:
            response_type = 'both'
        if not (0.0 <= polygon_detail_level <= 0.2):
            polygon_detail_level = 0.002
        max_results = max(1, min(max_results, 1000))
        return response_type, polygon_detail_level, max_results

    def get_predictor(self):
        if not hasattr(self, '_predictor'):
            logger.info('Creating FastSAM predictor')
            self._predictor = SAMPredictor('FastSAM')
        return self._predictor

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        inference_start_time = time.time()
        response_type, polygon_detail_level, max_results = self.get_runtime_config(**kwargs)

        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')
        polygon_from_name = None
        try:
            polygon_from_name, _, _ = self.get_first_tag_occurence('PolygonLabels', 'Image')
        except Exception:
            polygon_from_name = from_name

        # No interaction yet: FastSAM everything/auto
        if not context or not context.get('result'):
            img_url = tasks[0]['data'][value]
            predictor = self.get_predictor()
            predictor_results = predictor.predict(
                img_path=img_url,
                point_coords=None,
                point_labels=None,
                input_box=None,
                task=tasks[0]
            )

            if not predictor_results['masks']:
                return []

            # Determine image width/height
            image_width = tasks[0].get('meta', {}).get('image_width') or None
            image_height = tasks[0].get('meta', {}).get('image_height') or None
            image_path = None
            if image_width is None or image_height is None or response_type in ['polygon', 'both']:
                # Need local image for dimensions and mean intensity
                try:
                    hostname, access_token = get_credentials_for_task(tasks[0])
                    image_path = self.get_local_path(img_url, ls_host=hostname, ls_access_token=access_token, task_id=tasks[0].get('id'))
                    image = cv2.imread(image_path)
                    if image is None:
                        return []
                    image_height, image_width = image.shape[:2]
                except Exception:
                    return []

            # Build results
            predictions = self.get_results(
                masks=predictor_results['masks'],
                probs=predictor_results['probs'],
                width=image_width,
                height=image_height,
                from_name=from_name,
                to_name=to_name,
                label='Auto',
                polygon_from_name=polygon_from_name,
                response_type=response_type,
                polygon_detail_level=polygon_detail_level,
                max_results=max_results,
                image_path=image_path
            )
            logger.debug(f"Total inference time (AMG): {time.time() - inference_start_time:.4f}s")
            return predictions

        # Interactive prompts
        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']
        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None
        for ctx in context['result']:
            x = ctx['value']['x'] * image_width / 100
            y = ctx['value']['y'] * image_height / 100
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]
            if ctx_type == 'keypointlabels':
                point_labels.append(int(ctx.get('is_positive', 0)))
                point_coords.append([int(x), int(y)])
            elif ctx_type == 'rectanglelabels':
                box_width = ctx['value']['width'] * image_width / 100
                box_height = ctx['value']['height'] * image_height / 100
                input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]

        img_path = tasks[0]['data'][value]
        predictor = self.get_predictor()
        predictor_results = predictor.predict(
            img_path=img_path,
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            input_box=input_box,
            task=tasks[0]
        )

        # Local path for mean intensity when polygons requested
        local_img_path = None
        if response_type in ['polygon', 'both']:
            try:
                hostname, access_token = get_credentials_for_task(tasks[0])
                local_img_path = self.get_local_path(img_path, ls_host=hostname, ls_access_token=access_token, task_id=tasks[0].get('id'))
            except Exception:
                local_img_path = None

        predictions = self.get_results(
            masks=predictor_results['masks'],
            probs=predictor_results['probs'],
            width=image_width,
            height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=selected_label or 'Auto',
            polygon_from_name=polygon_from_name,
            response_type=response_type,
            polygon_detail_level=polygon_detail_level,
            max_results=max_results,
            image_path=local_img_path
        )

        logger.debug(f"Total inference time: {time.time() - inference_start_time:.4f}s")
        return predictions

    def calculate_mean_intensity(self, image_path, polygon_points, width, height):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            pixel_coords = []
            for i in range(0, len(polygon_points), 2):
                x_percent = polygon_points[i]
                y_percent = polygon_points[i + 1]
                x_pixel = int((x_percent / 100) * width)
                y_pixel = int((y_percent / 100) * height)
                pixel_coords.append([x_pixel, y_pixel])
            if len(pixel_coords) < 3:
                return None
            x_coords = [coord[0] for coord in pixel_coords]
            y_coords = [coord[1] for coord in pixel_coords]
            rr, cc = skimage_polygon(y_coords, x_coords, shape=image.shape)
            valid = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
            rr = rr[valid]
            cc = cc[valid]
            if len(rr) == 0:
                return None
            polygon_pixels = image[rr, cc]
            return float(np.mean(polygon_pixels))
        except Exception:
            return None

    def extract_largest_contour_polygon(self, mask, width, height, detail_level=None):
        if detail_level is None:
            detail_level = POLYGON_DETAIL_LEVEL
        binary_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = detail_level * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        polygon_points = []
        for point in simplified_contour:
            x, y = point[0]
            x_percent = (x / width) * 100
            y_percent = (y / height) * 100
            polygon_points.extend([x_percent, y_percent])
        return polygon_points

    def get_results(self, masks, probs, width, height, from_name, to_name, label, polygon_from_name=None, response_type=None, polygon_detail_level=None, max_results=None, image_path=None):
        if response_type is None:
            response_type = RESPONSE_TYPE
        if polygon_detail_level is None:
            polygon_detail_level = POLYGON_DETAIL_LEVEL
        if max_results is None:
            max_results = MAX_RESULTS
        results = []
        total_prob = 0
        result_count = 0
        for mask, prob in zip(masks, probs):
            total_prob += prob
            if result_count >= max_results:
                break
            if response_type in ['brush', 'both']:
                label_id = str(uuid4())[:4]
                mask_rle = (mask * 255).astype(np.uint8)
                rle = brush.mask2rle(mask_rle)
                results.append({
                    'id': label_id,
                    'from_name': from_name,
                    'to_name': to_name,
                    'original_width': width,
                    'original_height': height,
                    'image_rotation': 0,
                    'value': {
                        'format': 'rle',
                        'rle': rle,
                        'brushlabels': [label],
                    },
                    'score': prob,
                    'type': 'brushlabels',
                    'readonly': False
                })
            if response_type in ['polygon', 'both'] and polygon_from_name:
                polygon_points = self.extract_largest_contour_polygon(mask, width, height, polygon_detail_level)
                if polygon_points and len(polygon_points) >= 6:
                    polygon_label_id = str(uuid4())[:4]
                    points_pairs = []
                    for j in range(0, len(polygon_points), 2):
                        points_pairs.append([polygon_points[j], polygon_points[j+1]])
                    mean_intensity = None
                    if image_path:
                        mean_intensity = self.calculate_mean_intensity(image_path, polygon_points, width, height)
                    polygon_result = {
                        'id': polygon_label_id,
                        'from_name': polygon_from_name,
                        'to_name': to_name,
                        'original_width': width,
                        'original_height': height,
                        'image_rotation': 0,
                        'value': {
                            'points': points_pairs,
                            'polygonlabels': [label],
                        },
                        'score': prob,
                        'type': 'polygon',
                        'readonly': False
                    }
                    results.append(polygon_result)
                    if mean_intensity is not None:
                        textarea_from_name = None
                        try:
                            textarea_from_name, _, _ = self.get_first_tag_occurence('TextArea', 'Image')
                        except Exception:
                            textarea_from_name = 'mean_intensity'
                        results.append({
                            'id': polygon_label_id,
                            'from_name': textarea_from_name,
                            'to_name': to_name,
                            'original_width': width,
                            'original_height': height,
                            'image_rotation': 0,
                            'value': {
                                'points': points_pairs,
                                'text': [f"{mean_intensity:.2f}"]
                            },
                            'score': prob,
                            'type': 'textarea',
                            'readonly': False
                        })
            result_count += 1
        return [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_prob / max(len(results), 1)
        }]


