import torch
import numpy as np
import os
import sys
import pathlib
from typing import List, Dict, Optional
import json
import re
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from PIL import Image
import cv2
from skimage.draw import polygon as skimage_polygon

ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# SAM2 model choice (similar to original SAM_CHOICE)
SAM2_CHOICE = os.getenv('SAM2_CHOICE', 'tiny')  # options: tiny, small, base, large

# Model configurations based on SAM2_CHOICE
SAM2_MODEL_CONFIGS = {
    'tiny': 'configs/sam2.1/sam2.1_hiera_t.yaml',
    'small': 'configs/sam2.1/sam2.1_hiera_s.yaml',
    'base': 'configs/sam2.1/sam2.1_hiera_b.yaml',
    'large': 'configs/sam2.1/sam2.1_hiera_l.yaml'
}

SAM2_MODEL_CHECKPOINTS = {
    'tiny': 'sam2.1_hiera_tiny.pt',
    'small': 'sam2.1_hiera_small.pt',
    'base': 'sam2.1_hiera_base_plus.pt',
    'large': 'sam2.1_hiera_large.pt'
}

# Allow override via environment variables
MODEL_CONFIG = os.getenv('MODEL_CONFIG', SAM2_MODEL_CONFIGS[SAM2_CHOICE])
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', SAM2_MODEL_CHECKPOINTS[SAM2_CHOICE])

DEVICE = os.getenv('DEVICE', 'cuda')

# Runtime output/capabilities (ported from ls-ml-backend-SAM)
RESPONSE_TYPE = os.environ.get('RESPONSE_TYPE', 'both')  # brush | polygon | both
POLYGON_DETAIL_LEVEL = float(os.environ.get('POLYGON_DETAIL_LEVEL', '0.002'))  # 0.001-0.01
MAX_RESULTS = int(os.environ.get('MAX_RESULTS', '10'))

# AMG preannotation (SAM2 AutomaticMaskGenerator)
SAM_PREANNOTATE = os.environ.get('SAM_PREANNOTATE', '0') in ('1', 'true', 'True')
SAM_AMG_POINTS_PER_SIDE = int(os.environ.get('SAM_AMG_POINTS_PER_SIDE', '32'))
SAM_AMG_PRED_IOU_THRESH = float(os.environ.get('SAM_AMG_PRED_IOU_THRESH', '0.86'))
SAM_AMG_STABILITY_SCORE_THRESH = float(os.environ.get('SAM_AMG_STABILITY_SCORE_THRESH', '0.95'))
SAM_AMG_MIN_MASK_REGION_AREA = int(os.environ.get('SAM_AMG_MIN_MASK_REGION_AREA', '50'))
SAM_AMG_CROP_N_LAYERS = int(os.environ.get('SAM_AMG_CROP_N_LAYERS', '0'))
SAM_AMG_NMS_IOU_THRESH = float(os.environ.get('SAM_AMG_NMS_IOU_THRESH', '0.7'))

if DEVICE == 'cuda':
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# build path to the model checkpoint
sam2_checkpoint = str(os.path.join(ROOT_DIR, "checkpoints", MODEL_CHECKPOINT))

sam2_model = build_sam2(MODEL_CONFIG, sam2_checkpoint, device=DEVICE)

predictor = SAM2ImagePredictor(sam2_model)


def _compute_iou_between_masks(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a_bin = (a.astype(np.uint8) > 0)
    b_bin = (b.astype(np.uint8) > 0)
    intersection = np.logical_and(a_bin, b_bin).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _filter_overlapping_masks_by_iou(masks_data: List[Dict], iou_threshold: float) -> List[Dict]:
    if not masks_data:
        return masks_data
    filtered: List[Dict] = []
    for candidate in masks_data:
        cand_mask = candidate.get('segmentation')
        if cand_mask is None:
            continue
        is_overlapping = False
        for kept in filtered:
            kept_mask = kept.get('segmentation')
            iou = _compute_iou_between_masks(cand_mask, kept_mask)
            if iou >= iou_threshold:
                is_overlapping = True
                break
        if not is_overlapping:
            filtered.append(candidate)
    return filtered


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def setup(self):
        """Read connection-level overrides from Label Studio extra_params.

        Supports either a JSON object or simple KEY=VALUE pairs separated by newlines or '&'.
        Stores the connection override for preannotation so it can take precedence over
        environment defaults during predict calls.
        """
        # Default: no connection-level override (use env var)
        self._conn_preannotate = None
        self._conn_overrides: Dict[str, object] = {}

        # Read raw stored value; may be JSON or plaintext
        try:
            raw_extra = self.get('extra_params')
        except Exception:
            raw_extra = None

        extra: Dict[str, object] = {}
        if isinstance(raw_extra, dict):
            extra = raw_extra
        elif isinstance(raw_extra, str):
            # Try JSON first
            try:
                parsed = json.loads(raw_extra)
                if isinstance(parsed, dict):
                    extra = parsed
                else:
                    extra = {}
            except Exception:
                # Fallback: parse KEY=VALUE pairs split by newlines or '&'
                try:
                    pairs = re.split(r'[&\n]', raw_extra)
                    for pair in pairs:
                        pair = pair.strip()
                        if '=' in pair:
                            k, v = pair.split('=', 1)
                            extra[k.strip()] = v.strip()
                except Exception:
                    extra = {}

        def parse_bool(value, default=None):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            return str(value).strip().lower() in ('1', 'true', 'yes', 'on')

        # Save connection-level overrides
        self._conn_overrides = extra
        # Save dedicated flag (can be True/False or None if not provided)
        self._conn_preannotate = parse_bool(extra.get('SAM_PREANNOTATE'), None)

    def _resolve_config(self, **kwargs):
        """Resolve runtime configuration with precedence:
        kwargs > connection-level extra_params > env defaults.
        Returns a dict with typed values.
        """
        extra = getattr(self, '_conn_overrides', {}) or {}

        def parse_bool(value, default):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            return str(value).strip().lower() in ('1', 'true', 'yes', 'on')

        def parse_int(value, default):
            try:
                return int(value)
            except Exception:
                return default

        def parse_float(value, default):
            try:
                return float(value)
            except Exception:
                return default

        # Helper that checks lowercase and uppercase keys in overrides
        def ov(key, default=None):
            return extra.get(key, extra.get(key.lower(), default))

        # Preannotation flag
        preannotate = kwargs.get('preannotate', None)
        if isinstance(preannotate, str):
            preannotate = preannotate.strip().lower() in ('1', 'true', 'yes', 'on')
        if preannotate is None:
            preannotate_flag = parse_bool(ov('SAM_PREANNOTATE', None), SAM_PREANNOTATE)
        else:
            preannotate_flag = bool(preannotate)

        # Output controls
        response_type = kwargs.get('response_type') or ov('RESPONSE_TYPE', RESPONSE_TYPE)
        if response_type not in ['brush', 'polygon', 'both']:
            response_type = 'both'

        polygon_detail_level = parse_float(
            kwargs.get('polygon_detail_level') or ov('POLYGON_DETAIL_LEVEL', POLYGON_DETAIL_LEVEL),
            POLYGON_DETAIL_LEVEL,
        )
        if not (0.0 <= polygon_detail_level <= 0.2):
            polygon_detail_level = POLYGON_DETAIL_LEVEL

        max_results = parse_int(
            kwargs.get('max_results') or ov('MAX_RESULTS', MAX_RESULTS),
            MAX_RESULTS,
        )
        max_results = max(1, min(max_results, 1000))

        # AMG parameters
        points_per_side = parse_int(ov('SAM_AMG_POINTS_PER_SIDE', SAM_AMG_POINTS_PER_SIDE), SAM_AMG_POINTS_PER_SIDE)
        pred_iou_thresh = parse_float(ov('SAM_AMG_PRED_IOU_THRESH', SAM_AMG_PRED_IOU_THRESH), SAM_AMG_PRED_IOU_THRESH)
        stability_score_thresh = parse_float(
            ov('SAM_AMG_STABILITY_SCORE_THRESH', SAM_AMG_STABILITY_SCORE_THRESH),
            SAM_AMG_STABILITY_SCORE_THRESH,
        )
        min_mask_region_area = parse_int(
            ov('SAM_AMG_MIN_MASK_REGION_AREA', SAM_AMG_MIN_MASK_REGION_AREA),
            SAM_AMG_MIN_MASK_REGION_AREA,
        )
        crop_n_layers = parse_int(ov('SAM_AMG_CROP_N_LAYERS', SAM_AMG_CROP_N_LAYERS), SAM_AMG_CROP_N_LAYERS)
        nms_iou_thresh = parse_float(ov('SAM_AMG_NMS_IOU_THRESH', SAM_AMG_NMS_IOU_THRESH), SAM_AMG_NMS_IOU_THRESH)

        return {
            'preannotate': preannotate_flag,
            'response_type': response_type,
            'polygon_detail_level': polygon_detail_level,
            'max_results': max_results,
            'points_per_side': points_per_side,
            'pred_iou_thresh': pred_iou_thresh,
            'stability_score_thresh': stability_score_thresh,
            'min_mask_region_area': min_mask_region_area,
            'crop_n_layers': crop_n_layers,
            'nms_iou_thresh': nms_iou_thresh,
        }

    def get_results(self, masks, probs, width, height, from_name, to_name, label,
                    polygon_from_name: Optional[str] = None,
                    response_type: Optional[str] = None,
                    polygon_detail_level: Optional[float] = None,
                    max_results: Optional[int] = None,
                    image_path: Optional[str] = None):
        if response_type is None:
            response_type = RESPONSE_TYPE
        if polygon_detail_level is None:
            polygon_detail_level = POLYGON_DETAIL_LEVEL
        if max_results is None:
            max_results = MAX_RESULTS

        results = []
        total_prob = 0.0
        processed = 0

        for mask, prob in zip(masks, probs):
            if processed >= max_results:
                break
            total_prob += prob

            if response_type in ['brush', 'both']:
                label_id = str(uuid4())[:4]
                mask_rle = (mask.astype(np.uint8) * 255)
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

                # Optional: add a TextArea with mean intensity for the same region
                if image_path:
                    mean_intensity = self.calculate_mean_intensity_from_mask(image_path, mask.astype(np.uint8))
                    if mean_intensity is not None:
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
                                'text': [f"{mean_intensity:.2f}"]
                            },
                            'score': prob,
                            'type': 'textarea',
                            'readonly': False
                        })

            if response_type in ['polygon', 'both'] and polygon_from_name:
                polygon_points = self.extract_largest_contour_polygon(mask, width, height, polygon_detail_level)
                if polygon_points and len(polygon_points) >= 6:
                    poly_id = str(uuid4())[:4]
                    points_pairs = []
                    for i in range(0, len(polygon_points), 2):
                        points_pairs.append([polygon_points[i], polygon_points[i+1]])

                    polygon_result = {
                        'id': poly_id,
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

                    # Optional: add a TextArea with mean intensity for the same region
                    if image_path:
                        mean_intensity = self.calculate_mean_intensity(image_path, polygon_points, width, height)
                        if mean_intensity is not None:
                            textarea_from_name = None
                            try:
                                textarea_from_name, _, _ = self.get_first_tag_occurence('TextArea', 'Image')
                            except Exception:
                                textarea_from_name = 'mean_intensity'
                            results.append({
                                'id': poly_id,
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

            processed += 1

        return [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_prob / max(len(results), 1)
        }]

    def set_image(self, image_url, task: Optional[Dict]):
        access_token = None
        hostname = None
        try:
            # Optional middleware integration
            from org_api_middleware_v3 import get_credentials_for_task as _get_creds
            if task is not None:
                hostname, access_token, _ = _get_creds(task)
        except Exception:
            pass

        image_path = get_local_path(
            image_url,
            access_token=access_token,
            hostname=hostname,
            task_id=task.get('id') if task else None
        )
        # Simple embedding reuse: avoid resetting predictor if same image URL
        if getattr(self, '_last_image_url', None) == image_url:
            return
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        predictor.set_image(image)
        self._last_image_url = image_url

    def _sam_predict(self, img_url, point_coords=None, point_labels=None, input_box=None, task=None):
        self.set_image(img_url, task)
        point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        input_box = np.array(input_box, dtype=np.float32) if input_box else None

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            multimask_output=True
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        mask = masks[0, :, :].astype(np.uint8)
        prob = float(scores[0])
        # logits = logits[sorted_ind]
        return {
            'masks': [mask],
            'probs': [prob]
        }


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Returns predictions based on interactions or runs SAM2 AMG for preannotation."""

        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')

        # Try to resolve a polygon control if present
        polygon_from_name = None
        try:
            polygon_from_name, _, _ = self.get_first_tag_occurence('PolygonLabels', 'Image')
        except Exception:
            polygon_from_name = None

        # Preannotation path (no context yet)
        if not context or not context.get('result'):
            # Resolve full config for preannotation path
            cfg = self._resolve_config(**kwargs)
            if not cfg['preannotate']:
                return ModelResponse(predictions=[])

            img_url = tasks[0]['data'][value]
            access_token = None
            hostname = None
            try:
                from org_api_middleware_v3 import get_credentials_for_task as _get_creds
                hostname, access_token, _ = _get_creds(tasks[0])
            except Exception:
                pass

            local_img_path = get_local_path(
                img_url,
                access_token=access_token,
                hostname=hostname,
                task_id=tasks[0].get('id')
            )
            image = cv2.imread(local_img_path)
            if image is None:
                return ModelResponse(predictions=[])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            generator = SAM2AutomaticMaskGenerator(
                sam2_model,
                points_per_side=cfg['points_per_side'],
                pred_iou_thresh=cfg['pred_iou_thresh'],
                stability_score_thresh=cfg['stability_score_thresh'],
                min_mask_region_area=cfg['min_mask_region_area'],
                crop_n_layers=cfg['crop_n_layers'],
                output_mode='binary_mask',
                multimask_output=True,
            )

            try:
                masks_data = generator.generate(image)
            except Exception:
                masks_data = []

            # Convert and clean
            for md in masks_data:
                m = md.get('segmentation')
                if m is None:
                    continue
                bin_mask = (m.astype(np.uint8) > 0).astype(np.uint8)
                md['segmentation'] = bin_mask
                md['area'] = int(np.count_nonzero(bin_mask))

            # Sort by quality and area
            try:
                masks_data.sort(key=lambda d: (float(d.get('predicted_iou', 0.0)), int(d.get('area', 0))), reverse=True)
            except Exception:
                pass

            # IoU-based NMS
            try:
                masks_data = _filter_overlapping_masks_by_iou(masks_data, cfg['nms_iou_thresh'])
            except Exception:
                pass

            # Cap
            if cfg['max_results'] > 0:
                masks_data = masks_data[:cfg['max_results']]

            masks = []
            probs = []
            for md in masks_data:
                if md.get('segmentation') is not None:
                    masks.append(md['segmentation'].astype('uint8'))
                    probs.append(float(md.get('predicted_iou', 0.0)))

            if not masks:
                return ModelResponse(predictions=[])

            height, width = image.shape[:2]
            # Choose a label (first BrushLabel name)
            try:
                control = self.label_interface.get_control(from_name)
                label_names = list(control.labels_attrs.keys()) if control and control.labels_attrs else []
                selected_label = label_names[0] if label_names else 'Auto'
            except Exception:
                selected_label = 'Auto'

            predictions = self.get_results(
                masks=masks,
                probs=probs,
                width=width,
                height=height,
                from_name=from_name,
                to_name=to_name,
                label=selected_label,
                polygon_from_name=polygon_from_name,
                response_type=cfg['response_type'],
                polygon_detail_level=cfg['polygon_detail_level'],
                max_results=cfg['max_results'],
                image_path=local_img_path,
            )
            return ModelResponse(predictions=predictions)

        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        # collect context information
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

        print(f'Point coords are {point_coords}, point labels are {point_labels}, input box is {input_box}')

        img_url = tasks[0]['data'][value]
        predictor_results = self._sam_predict(
            img_url=img_url,
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            input_box=input_box,
            task=tasks[0]
        )

        # Resolve local image path for optional polygon-related calculations
        local_img_path = None
        try:
            from org_api_middleware_v3 import get_credentials_for_task as _get_creds
            hostname, access_token, _ = _get_creds(tasks[0])
            local_img_path = get_local_path(tasks[0]['data'][value], access_token=access_token, hostname=hostname, task_id=tasks[0].get('id'))
        except Exception:
            try:
                local_img_path = get_local_path(tasks[0]['data'][value], task_id=tasks[0].get('id'))
            except Exception:
                local_img_path = None

        predictions = self.get_results(
            masks=predictor_results['masks'],
            probs=predictor_results['probs'],
            width=image_width,
            height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=selected_label,
            polygon_from_name=polygon_from_name,
            response_type=RESPONSE_TYPE,
            polygon_detail_level=POLYGON_DETAIL_LEVEL,
            max_results=MAX_RESULTS,
            image_path=local_img_path,
        )
        
        return ModelResponse(predictions=predictions)

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
            x_coords = [pt[0] for pt in pixel_coords]
            y_coords = [pt[1] for pt in pixel_coords]
            rr, cc = skimage_polygon(y_coords, x_coords, shape=image.shape)
            valid = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
            rr = rr[valid]
            cc = cc[valid]
            if len(rr) == 0:
                return None
            return float(np.mean(image[rr, cc]))
        except Exception:
            return None

    def calculate_mean_intensity_from_mask(self, image_path, binary_mask):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            # Apply the binary mask to get only the pixels within the mask
            masked_pixels = image[binary_mask > 0]
            if len(masked_pixels) == 0:
                return None
            return float(np.mean(masked_pixels))
        except Exception:
            return None

    def extract_largest_contour_polygon(self, mask, width, height, detail_level=None):
        if detail_level is None:
            detail_level = POLYGON_DETAIL_LEVEL
        binary_mask = (mask.astype(np.uint8) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        epsilon = detail_level * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        polygon_points = []
        for pt in simplified:
            x, y = pt[0]
            x_percent = (x / width) * 100
            y_percent = (y / height) * 100
            polygon_points.extend([x_percent, y_percent])
        return polygon_points
