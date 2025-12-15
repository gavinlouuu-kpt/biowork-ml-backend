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
from sam_predictor import SAMPredictor, get_credentials_for_task
from label_studio_ml.model import LabelStudioMLBase
from PIL import Image, ImageOps

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
        
        # DEBUG: Log the environment variable values
        logger.debug(f"DEBUG: get_runtime_config - MAX_RESULTS env={MAX_RESULTS}, RESPONSE_TYPE env={RESPONSE_TYPE}")
        
        if response_type not in ['brush', 'polygon', 'both']:
            response_type = 'both'
        if not (0.0 <= polygon_detail_level <= 0.2):
            polygon_detail_level = 0.002
        max_results = max(1, min(max_results, 1000))
        
        # DEBUG: Log the final computed values
        logger.debug(f"DEBUG: get_runtime_config - final max_results={max_results}, response_type={response_type}")
        
        return response_type, polygon_detail_level, max_results

    def get_predictor(self):
        if not hasattr(self, '_predictor'):
            logger.info('Creating FastSAM predictor')
            self._predictor = SAMPredictor('FastSAM')
        return self._predictor

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        inference_start_time = time.time()
        response_type, polygon_detail_level, max_results = self.get_runtime_config(**kwargs)
        
        # DEBUG: Log the runtime configuration
        logger.debug(f"DEBUG: predict called with max_results={max_results}, response_type={response_type}")
        
        # Discover controls; prefer brush if present, but support polygon-only configs
        try:
            from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')
        except Exception:
            from_name = to_name = value = None

        polygon_from_name = None
        polygon_value = None
        polygon_to_name = None
        try:
            polygon_from_name, polygon_to_name, polygon_value = self.get_first_tag_occurence('PolygonLabels', 'Image')
        except Exception:
            polygon_from_name = from_name
            polygon_to_name = to_name

        # If no brush tag exists, fall back to polygon for image value/to_name
        if value is None and polygon_value is not None:
            value = polygon_value
            to_name = polygon_to_name or to_name or 'image'

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
                to_name=to_name or polygon_to_name,
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

        # Resolve EXIF-corrected image dimensions to convert percents → pixels reliably
        img_url = tasks[0]['data'][value]
        true_width, true_height = image_width, image_height
        try:
            hostname, access_token = get_credentials_for_task(tasks[0])
            _local_img_for_dims = self.get_local_path(
                img_url,
                ls_host=hostname,
                ls_access_token=access_token,
                task_id=tasks[0].get('id')
            )
            _pil = Image.open(_local_img_for_dims)
            _pil = ImageOps.exif_transpose(_pil)
            true_width, true_height = _pil.size
        except Exception:
            # Fall back to context dims if local image can't be opened
            true_width, true_height = image_width, image_height
        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None
        for ctx in context['result']:
            # Convert Label Studio percentage coords using EXIF-corrected dimensions
            x = ctx['value']['x'] * true_width / 100
            y = ctx['value']['y'] * true_height / 100
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]
            if ctx_type == 'keypointlabels':
                xi, yi = int(x), int(y)
                # Strict validation: if any point is outside, ignore the interaction
                if not (0 <= xi < true_width and 0 <= yi < true_height):
                    logger.debug(
                        f"Ignored OOB click (x={xi}, y={yi}) for task={tasks[0].get('id')} within image={true_width}x{true_height}"
                    )
                    return []
                point_labels.append(int(ctx.get('is_positive', 0)))
                point_coords.append([xi, yi])
            elif ctx_type == 'rectanglelabels':
                box_width = ctx['value']['width'] * true_width / 100
                box_height = ctx['value']['height'] * true_height / 100
                x1, y1 = int(x), int(y)
                x2, y2 = int(box_width + x), int(box_height + y)
                # Validate rectangle is fully inside and non-empty
                if not (0 <= x1 < x2 <= true_width and 0 <= y1 < y2 <= true_height):
                    logger.debug(
                        f"Ignored OOB rectangle ({x1},{y1},{x2},{y2}) for task={tasks[0].get('id')} within image={true_width}x{true_height}"
                    )
                    return []
                input_box = [x1, y1, x2, y2]

        img_path = img_url
        predictor = self.get_predictor()
        predictor_results = predictor.predict(
            img_path=img_path,
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            input_box=input_box,
            task=tasks[0]
        )

        # Local path for mean intensity for polygons/brush
        local_img_path = None
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

    def _load_image_for_intensity(self, image_path):
        """
        Load image for intensity computation, normalizing to a 3-channel BGR image.

        - Grayscale images are converted with cv2.COLOR_GRAY2BGR so that
          downstream code always sees three channels where grayscale corresponds
          to r=g=b.
        - BGRA images are converted to BGR.
        """
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        # Grayscale -> BGR (r=g=b)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3:
            # BGRA -> BGR
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return image
        return None

    def _compute_mean_intensity_channels(self, image, mask_bool):
        """
        Compute RGB channel means for provided boolean mask.

        All images are expected to be 3-channel BGR; grayscale input is handled
        by having r=g=b for the region. The returned dict encodes only RGB
        channels; grayscale can be inferred when r≈g≈b.
        """
        if mask_bool.sum() == 0:
            return {'r': 0.0, 'g': 0.0, 'b': 0.0}

        # Defensive handling in case a 2D image slips through
        if image.ndim == 2:
            mean_val = float(np.mean(image[mask_bool]))
            return {'r': mean_val, 'g': mean_val, 'b': mean_val}

        if image.ndim == 3:
            if image.shape[2] == 1:
                mean_val = float(np.mean(image[mask_bool]))
                return {'r': mean_val, 'g': mean_val, 'b': mean_val}
            b, g, r = cv2.split(image)
            mean_r = float(np.mean(r[mask_bool]))
            mean_g = float(np.mean(g[mask_bool]))
            mean_b = float(np.mean(b[mask_bool]))
            return {'r': mean_r, 'g': mean_g, 'b': mean_b}

        # Fallback: unexpected image shape
        return {'r': 0.0, 'g': 0.0, 'b': 0.0}

    def _format_mean_intensity_text(self, mean_dict):
        """
        Return a deterministic RGB-only string for textarea storage.

        The backend now emits only per-channel RGB means. Grayscale / black and
        white regions are represented by r=g=b; no separate gray channel is
        stored in the text.
        """
        if mean_dict is None:
            return None

        # Legacy float: treat as grayscale encoded as r=g=b=value.
        if isinstance(mean_dict, (int, float)):
            val = float(mean_dict)
            r = g = b = val
            return f"r={r:.2f}; g={g:.2f}; b={b:.2f}"

        if isinstance(mean_dict, dict):
            r = mean_dict.get('r')
            g = mean_dict.get('g')
            b = mean_dict.get('b')

            # Legacy gray-only dict: use gray as r=g=b if RGB are missing.
            if r is None and g is None and b is None:
                gray = mean_dict.get('gray')
                if gray is None:
                    return None
                r = g = b = float(gray)
            else:
                r = 0.0 if r is None else float(r)
                g = 0.0 if g is None else float(g)
                b = 0.0 if b is None else float(b)

            return f"r={r:.2f}; g={g:.2f}; b={b:.2f}"

        # Unexpected type: best-effort float coercion
        try:
            val = float(mean_dict)
        except Exception:
            return None
        r = g = b = val
        return f"r={r:.2f}; g={g:.2f}; b={b:.2f}"

    def calculate_mean_intensity(self, image_path, polygon_points, width, height):
        try:
            image = self._load_image_for_intensity(image_path)
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
            mask = np.zeros(image.shape[:2], dtype=bool)
            x_coords = [coord[0] for coord in pixel_coords]
            y_coords = [coord[1] for coord in pixel_coords]
            rr, cc = skimage_polygon(y_coords, x_coords, shape=mask.shape)
            valid = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
            mask[rr[valid], cc[valid]] = True
            if not mask.any():
                return None
            return self._compute_mean_intensity_channels(image, mask)
        except Exception:
            return None

    def calculate_mean_intensity_from_mask(self, image_path, binary_mask):
        try:
            image = self._load_image_for_intensity(image_path)
            if image is None:
                return None
            mask_bool = (binary_mask.astype(np.uint8) > 0)
            if mask_bool.shape != image.shape[:2]:
                mask_bool = cv2.resize(mask_bool.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not mask_bool.any():
                return None
            return self._compute_mean_intensity_channels(image, mask_bool)
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
        
        # DEBUG: Log the max_results value and number of input masks
        logger.debug(f"DEBUG: get_results called with max_results={max_results}, input masks count={len(masks)}")
        
        results = []
        total_prob = 0
        result_count = 0

        # Resolve valid labels from the controls to avoid "No label" in UI
        try:
            brush_control = self.label_interface.get_control(from_name)
            brush_label_names = list(brush_control.labels_attrs.keys()) if brush_control and brush_control.labels_attrs else []
            brush_label_value = brush_label_names[0] if brush_label_names else (label or 'Auto')
        except Exception:
            brush_label_value = label or 'Auto'

        try:
            polygon_control = self.label_interface.get_control(polygon_from_name) if polygon_from_name else None
            polygon_label_names = list(polygon_control.labels_attrs.keys()) if polygon_control and polygon_control.labels_attrs else []
            polygon_label_value = polygon_label_names[0] if polygon_label_names else (label or 'Auto')
        except Exception:
            polygon_label_value = label or 'Auto'
            
        # DEBUG: Log before processing masks
        logger.debug(f"DEBUG: Starting mask processing, total masks available: {len(masks)}")
            
        for mask, prob in zip(masks, probs):
            total_prob += prob
            if result_count >= max_results:
                # DEBUG: Log when we hit the max_results limit
                logger.debug(f"DEBUG: Reached max_results limit ({max_results}), stopping processing")
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
                        'brushlabels': [brush_label_value],
                    },
                    'score': prob,
                    'type': 'brushlabels',
                    'readonly': False
                })
                if image_path:
                    mean_intensity = self.calculate_mean_intensity_from_mask(image_path, mask.astype(np.uint8))
                    mean_text = self._format_mean_intensity_text(mean_intensity)
                    if mean_text:
                        textarea_from_name = None
                        try:
                            textarea_from_name, _, _ = self.get_first_tag_occurence('TextArea', 'Image')
                        except Exception:
                            textarea_from_name = 'mean_intensity'
                        results.append({
                            'id': label_id,
                            'from_name': textarea_from_name,
                            'to_name': to_name,
                            'original_width': width,
                            'original_height': height,
                            'image_rotation': 0,
                            'value': {
                                'format': 'rle',
                                'rle': rle,
                                'text': [mean_text]
                            },
                            'score': prob,
                            'type': 'textarea',
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
                            'polygonlabels': [polygon_label_value],
                        },
                        'score': prob,
                        'type': 'polygon',
                        'readonly': False
                    }
                    results.append(polygon_result)
                    mean_text = self._format_mean_intensity_text(mean_intensity)
                    if mean_text:
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
                                'text': [mean_text]
                            },
                            'score': prob,
                            'type': 'textarea',
                            'readonly': False
                        })
            result_count += 1
            
            # DEBUG: Log progress every 50 masks
            if result_count % 50 == 0:
                logger.debug(f"DEBUG: Processed {result_count} masks so far")
        
        # DEBUG: Log final results count
        logger.debug(f"DEBUG: Final results count: {len(results)}, total masks processed: {result_count}")
        
        return [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_prob / max(len(results), 1)
        }]


