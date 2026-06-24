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
from label_studio_ml.exceptions import ValidationError
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

    def _empty_prediction_response(self) -> ModelResponse:
        """Return a non-error empty prediction envelope.

        Label Studio's model test endpoint treats `predictions=[]` as an error.
        Returning one prediction with an empty `result` keeps interactive SAM
        behavior intact while allowing connectivity tests to pass.
        """
        return ModelResponse(predictions=[{
            'result': [],
            'model_version': self.get('model_version'),
            'score': 0.0,
        }])

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

    def _prompt_validation_error(self, message: str):
        raise ValidationError(f"Invalid SAM2 prompt payload: {message}")

    def _to_finite_float(self, value, path: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            self._prompt_validation_error(f"{path} must be a number")
        if not np.isfinite(number):
            self._prompt_validation_error(f"{path} must be a finite number")
        return number

    def _parse_boolish(self, value, path: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ('1', 'true', 'yes', 'positive', 'foreground'):
                return True
            if normalized in ('0', 'false', 'no', 'negative', 'background'):
                return False
        self._prompt_validation_error(f"{path} must be a boolean or positive/negative value")

    def _coordinate_system(self, value, path: str) -> str:
        if value is None:
            return 'percent'
        normalized = str(value).strip().lower()
        if normalized in ('percent', 'percentage', 'perc'):
            return 'percent'
        if normalized in ('pixel', 'pixels', 'px'):
            return 'pixel'
        if normalized in ('normalized', 'relative', 'ratio'):
            return 'normalized'
        self._prompt_validation_error(
            f"{path} must be one of: percent, pixel, normalized"
        )

    def _dimension_pair_from_context(self, context: Dict, payload: Dict) -> tuple:
        payload = payload or {}
        context = context or {}

        def first_value(*keys):
            for source in (payload, context):
                for key in keys:
                    value = source.get(key)
                    if value is not None:
                        return value
            return None

        def payload_width_height_are_box_geometry() -> bool:
            return (
                self._prompt_item_kind(payload) == 'box'
                and all(key in payload for key in ('x', 'y', 'width', 'height'))
                and not any(key in payload for key in ('original_width', 'original_height', 'image_width', 'image_height'))
            )

        width = first_value('original_width', 'image_width')
        height = first_value('original_height', 'image_height')
        if width is None:
            width = context.get('width')
        if height is None:
            height = context.get('height')
        if not payload_width_height_are_box_geometry():
            if width is None:
                width = payload.get('width')
            if height is None:
                height = payload.get('height')

        width = self._to_finite_float(width, 'prompts.original_width')
        height = self._to_finite_float(height, 'prompts.original_height')
        if width <= 0 or height <= 0:
            self._prompt_validation_error('image dimensions must be greater than zero')
        return int(round(width)), int(round(height))

    def _coord_to_pixels(self, value, size: int, coordinate_system: str, path: str) -> float:
        number = self._to_finite_float(value, path)
        if coordinate_system == 'percent':
            if number < 0 or number > 100:
                self._prompt_validation_error(f"{path} percent value must be between 0 and 100")
            return number * size / 100.0
        if coordinate_system == 'normalized':
            if number < 0 or number > 1:
                self._prompt_validation_error(f"{path} normalized value must be between 0 and 1")
            return number * size
        if number < 0 or number > size:
            self._prompt_validation_error(f"{path} pixel value must be between 0 and {size}")
        return number

    def _extract_prompt_label(self, value, selected_label: Optional[str]) -> Optional[str]:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]
        return selected_label

    def _prompt_item_kind(self, prompt: Dict) -> str:
        for key in ('prompt_type', 'promptType', 'kind', 'type'):
            value = prompt.get(key)
            if not isinstance(value, str):
                continue
            normalized = value.strip().lower()
            if normalized in ('box', 'bbox', 'rect', 'rectangle', 'rectanglelabels'):
                return 'box'
            if normalized in ('point', 'points', 'keypoint', 'keypointlabels'):
                return 'point'
        if prompt.get('box') is not None or all(key in prompt for key in ('x1', 'y1', 'x2', 'y2')):
            return 'box'
        if all(key in prompt for key in ('x', 'y', 'width', 'height')) and not any(
            key in prompt for key in ('is_positive', 'positive', 'polarity')
        ):
            return 'box'
        return 'point'

    def _point_label_value(self, prompt: Dict, path: str) -> int:
        if 'is_positive' in prompt:
            return 1 if self._parse_boolish(prompt.get('is_positive'), f'{path}.is_positive') else 0
        if 'positive' in prompt:
            return 1 if self._parse_boolish(prompt.get('positive'), f'{path}.positive') else 0
        if 'polarity' in prompt:
            return 1 if self._parse_boolish(prompt.get('polarity'), f'{path}.polarity') else 0
        if prompt.get('label') in (0, 1, '0', '1'):
            return int(prompt.get('label'))
        if 'type' in prompt:
            prompt_type = prompt.get('type')
            if isinstance(prompt_type, str) and prompt_type.strip().lower() in ('point', 'points', 'keypoint', 'keypointlabels'):
                return 1
            return 1 if self._parse_boolish(prompt_type, f'{path}.type') else 0
        return 1

    def _point_xy(self, prompt: Dict, image_width: int, image_height: int, default_coordinate_system: str, path: str):
        coordinate_system = self._coordinate_system(
            prompt.get('coordinate_system', default_coordinate_system),
            f'{path}.coordinate_system',
        )
        x = prompt.get('x')
        y = prompt.get('y')
        if x is None or y is None:
            coords = prompt.get('point', prompt.get('coordinates'))
            if isinstance(coords, (list, tuple)) and len(coords) == 2:
                x, y = coords
        if x is None or y is None:
            self._prompt_validation_error(f'{path} must include x/y or point coordinates')
        return [
            self._coord_to_pixels(x, image_width, coordinate_system, f'{path}.x'),
            self._coord_to_pixels(y, image_height, coordinate_system, f'{path}.y'),
        ]

    def _box_xyxy(self, prompt: Dict, image_width: int, image_height: int, default_coordinate_system: str, path: str):
        coordinate_system = self._coordinate_system(
            prompt.get('coordinate_system', default_coordinate_system),
            f'{path}.coordinate_system',
        )
        rotation = prompt.get('rotation', 0)
        if rotation not in (0, 0.0, None):
            self._prompt_validation_error(f'{path}.rotation is not supported for SAM2 box prompts')

        raw_box = prompt.get('box')
        if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            x1, y1, x2, y2 = raw_box
            x1 = self._coord_to_pixels(x1, image_width, coordinate_system, f'{path}.box[0]')
            y1 = self._coord_to_pixels(y1, image_height, coordinate_system, f'{path}.box[1]')
            x2 = self._coord_to_pixels(x2, image_width, coordinate_system, f'{path}.box[2]')
            y2 = self._coord_to_pixels(y2, image_height, coordinate_system, f'{path}.box[3]')
        elif all(key in prompt for key in ('x1', 'y1', 'x2', 'y2')):
            x1 = self._coord_to_pixels(prompt.get('x1'), image_width, coordinate_system, f'{path}.x1')
            y1 = self._coord_to_pixels(prompt.get('y1'), image_height, coordinate_system, f'{path}.y1')
            x2 = self._coord_to_pixels(prompt.get('x2'), image_width, coordinate_system, f'{path}.x2')
            y2 = self._coord_to_pixels(prompt.get('y2'), image_height, coordinate_system, f'{path}.y2')
        elif all(key in prompt for key in ('x', 'y', 'width', 'height')):
            x1 = self._coord_to_pixels(prompt.get('x'), image_width, coordinate_system, f'{path}.x')
            y1 = self._coord_to_pixels(prompt.get('y'), image_height, coordinate_system, f'{path}.y')
            width = self._coord_to_pixels(prompt.get('width'), image_width, coordinate_system, f'{path}.width')
            height = self._coord_to_pixels(prompt.get('height'), image_height, coordinate_system, f'{path}.height')
            x2 = x1 + width
            y2 = y1 + height
        else:
            self._prompt_validation_error(f'{path} must include box, x1/y1/x2/y2, or x/y/width/height')

        if x2 <= x1 or y2 <= y1:
            self._prompt_validation_error(f'{path} must describe a box with positive width and height')
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            self._prompt_validation_error(f'{path} must be within the image bounds')
        return [x1, y1, x2, y2]

    def _combined_box(self, boxes: List[List[float]]):
        if not boxes:
            return None
        return [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]

    def _get_default_label(self, from_name: str) -> str:
        try:
            control = self.label_interface.get_control(from_name)
            label_names = list(control.labels_attrs.keys()) if control and control.labels_attrs else []
            if label_names:
                return label_names[0]
        except Exception:
            pass
        return 'Object'

    def _normalize_prompt_payload(self, context: Optional[Dict], default_label: str) -> Dict:
        if not isinstance(context, dict):
            return {
                'empty': True,
                'image_width': 0,
                'image_height': 0,
                'point_coords': [],
                'point_labels': [],
                'input_box': None,
                'selected_label': default_label,
            }

        if 'prompts' in context:
            return self._normalize_structured_prompts(context, context.get('prompts'), default_label)
        if 'prompt' in context:
            return self._normalize_structured_prompts(context, context.get('prompt'), default_label)
        return self._normalize_label_studio_results(context, default_label)

    def _normalize_structured_prompts(self, context: Dict, payload, default_label: str) -> Dict:
        if isinstance(payload, list):
            payload = {'items': payload}
        if not isinstance(payload, dict):
            self._prompt_validation_error('prompts must be an object')

        selected_label = self._extract_prompt_label(payload.get('label'), default_label)
        points = payload.get('points', [])
        boxes = payload.get('boxes', [])
        items = payload.get('items', [])
        if not isinstance(points, list):
            self._prompt_validation_error('prompts.points must be a list')
        if not isinstance(boxes, list):
            self._prompt_validation_error('prompts.boxes must be a list')
        if not isinstance(items, list):
            self._prompt_validation_error('prompts.items must be a list')
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                self._prompt_validation_error(f'prompts.items[{index}] must be an object')
            if self._prompt_item_kind(item) == 'box':
                boxes = boxes + [item]
            else:
                points = points + [item]

        has_nested_prompts = bool(points or boxes or items)
        has_point_shorthand = payload.get('point') is not None
        has_box_shorthand = payload.get('box') is not None
        has_inline_geometry = (
            all(key in payload for key in ('x', 'y'))
            or all(key in payload for key in ('x1', 'y1', 'x2', 'y2'))
        )
        if not has_nested_prompts and not has_point_shorthand and not has_box_shorthand and has_inline_geometry:
            if self._prompt_item_kind(payload) == 'box':
                boxes = boxes + [payload]
            else:
                points = points + [payload]

        if payload.get('point') is not None:
            point_prompt = {'point': payload.get('point')}
            for key in ('coordinate_system', 'is_positive', 'positive', 'polarity', 'type', 'label_name'):
                if key in payload:
                    point_prompt[key] = payload.get(key)
            points = points + [point_prompt]
        if payload.get('box') is not None:
            box_prompt = {'box': payload.get('box')}
            for key in ('coordinate_system', 'rotation', 'label_name'):
                if key in payload:
                    box_prompt[key] = payload.get(key)
            boxes = boxes + [box_prompt]

        if not points and not boxes:
            return {
                'empty': True,
                'image_width': 0,
                'image_height': 0,
                'point_coords': [],
                'point_labels': [],
                'input_box': None,
                'selected_label': selected_label,
            }

        image_width, image_height = self._dimension_pair_from_context(context, payload)
        coordinate_system = self._coordinate_system(
            payload.get('coordinate_system', context.get('coordinate_system')),
            'prompts.coordinate_system',
        )

        point_coords = []
        point_labels = []
        for index, point in enumerate(points):
            path = f'prompts.points[{index}]'
            if not isinstance(point, dict):
                self._prompt_validation_error(f'{path} must be an object')
            point_coords.append(self._point_xy(point, image_width, image_height, coordinate_system, path))
            point_labels.append(self._point_label_value(point, path))
            selected_label = self._extract_prompt_label(point.get('label_name'), selected_label)

        parsed_boxes = []
        for index, box in enumerate(boxes):
            path = f'prompts.boxes[{index}]'
            if not isinstance(box, dict):
                self._prompt_validation_error(f'{path} must be an object')
            parsed_boxes.append(self._box_xyxy(box, image_width, image_height, coordinate_system, path))
            selected_label = self._extract_prompt_label(box.get('label_name'), selected_label)

        if not point_coords and not parsed_boxes:
            return {
                'empty': True,
                'image_width': image_width,
                'image_height': image_height,
                'point_coords': [],
                'point_labels': [],
                'input_box': None,
                'selected_label': selected_label,
            }
        if point_labels and not any(label == 1 for label in point_labels) and not parsed_boxes:
            self._prompt_validation_error('at least one positive point or box prompt is required')

        return {
            'empty': False,
            'image_width': image_width,
            'image_height': image_height,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'input_box': self._combined_box(parsed_boxes),
            'selected_label': selected_label,
        }

    def _normalize_label_studio_results(self, context: Dict, default_label: str) -> Dict:
        results = context.get('result') or []
        if not isinstance(results, list):
            self._prompt_validation_error('context.result must be a list')
        if not results:
            return {
                'empty': True,
                'image_width': 0,
                'image_height': 0,
                'point_coords': [],
                'point_labels': [],
                'input_box': None,
                'selected_label': default_label,
            }

        image_width = int(round(self._to_finite_float(results[0].get('original_width'), 'context.result[0].original_width')))
        image_height = int(round(self._to_finite_float(results[0].get('original_height'), 'context.result[0].original_height')))
        if image_width <= 0 or image_height <= 0:
            self._prompt_validation_error('context.result image dimensions must be greater than zero')

        point_coords = []
        point_labels = []
        boxes = []
        selected_label = default_label

        for index, ctx in enumerate(results):
            if not isinstance(ctx, dict):
                self._prompt_validation_error(f'context.result[{index}] must be an object')
            value = ctx.get('value') or {}
            if not isinstance(value, dict):
                self._prompt_validation_error(f'context.result[{index}].value must be an object')
            ctx_type = ctx.get('type')
            if ctx_type not in ('keypointlabels', 'rectanglelabels'):
                continue
            label_values = value.get(ctx_type) or value.get('labels') or []
            selected_label = self._extract_prompt_label(label_values, selected_label)

            if ctx_type == 'keypointlabels':
                x = self._to_finite_float(value.get('x'), f'context.result[{index}].value.x') * image_width / 100.0
                y = self._to_finite_float(value.get('y'), f'context.result[{index}].value.y') * image_height / 100.0
                point_coords.append([x, y])
                point_labels.append(1 if self._parse_boolish(ctx.get('is_positive', True), f'context.result[{index}].is_positive') else 0)
            elif ctx_type == 'rectanglelabels':
                x = self._to_finite_float(value.get('x'), f'context.result[{index}].value.x') * image_width / 100.0
                y = self._to_finite_float(value.get('y'), f'context.result[{index}].value.y') * image_height / 100.0
                box_width = self._to_finite_float(value.get('width'), f'context.result[{index}].value.width') * image_width / 100.0
                box_height = self._to_finite_float(value.get('height'), f'context.result[{index}].value.height') * image_height / 100.0
                if box_width <= 0 or box_height <= 0:
                    self._prompt_validation_error(f'context.result[{index}] rectangle must have positive width and height')
                boxes.append([x, y, x + box_width, y + box_height])

        if not point_coords and not boxes:
            return {
                'empty': True,
                'image_width': image_width,
                'image_height': image_height,
                'point_coords': [],
                'point_labels': [],
                'input_box': None,
                'selected_label': selected_label,
            }
        if point_labels and not any(label == 1 for label in point_labels) and not boxes:
            self._prompt_validation_error('at least one positive point or box prompt is required')

        return {
            'empty': False,
            'image_width': image_width,
            'image_height': image_height,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'input_box': self._combined_box(boxes),
            'selected_label': selected_label,
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

            # Geometry in **pixels** for this mask, reused across all results for it
            geometry_meta = self._compute_mask_geometry(mask)

            if response_type in ['brush', 'both']:
                label_id = str(uuid4())[:4]
                mask_rle = (mask.astype(np.uint8) * 255)
                rle = brush.mask2rle(mask_rle)
                brush_result = {
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
                    'readonly': False,
                }

                # Merge geometry stats and RGB mean intensities into `meta`
                meta = dict(geometry_meta) if geometry_meta is not None else {}
                if image_path:
                    mean_intensity = self.calculate_mean_intensity_from_mask(image_path, mask.astype(np.uint8))
                    if isinstance(mean_intensity, dict):
                        try:
                            r_val = float(mean_intensity.get('r', 0.0))
                        except Exception:
                            r_val = 0.0
                        try:
                            g_val = float(mean_intensity.get('g', 0.0))
                        except Exception:
                            g_val = 0.0
                        try:
                            b_val = float(mean_intensity.get('b', 0.0))
                        except Exception:
                            b_val = 0.0
                        meta.update({
                            "mean_r": r_val,
                            "mean_g": g_val,
                            "mean_b": b_val,
                        })

                if meta:
                    brush_result['meta'] = meta

                results.append(brush_result)

            if response_type in ['polygon', 'both'] and polygon_from_name:
                polygon_points = self.extract_largest_contour_polygon(mask, width, height, polygon_detail_level)
                if polygon_points and len(polygon_points) >= 6:
                    poly_id = str(uuid4())[:4]
                    points_pairs = []
                    for i in range(0, len(polygon_points), 2):
                        points_pairs.append([polygon_points[i], polygon_points[i+1]])

                    mean_intensity = None
                    if image_path:
                        mean_intensity = self.calculate_mean_intensity(image_path, polygon_points, width, height)

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
                        'readonly': False,
                    }
                    # Merge geometry stats and RGB mean intensities into `meta`
                    meta = dict(geometry_meta) if geometry_meta is not None else {}
                    if isinstance(mean_intensity, dict):
                        try:
                            r_val = float(mean_intensity.get('r', 0.0))
                        except Exception:
                            r_val = 0.0
                        try:
                            g_val = float(mean_intensity.get('g', 0.0))
                        except Exception:
                            g_val = 0.0
                        try:
                            b_val = float(mean_intensity.get('b', 0.0))
                        except Exception:
                            b_val = 0.0
                        meta.update({
                            "mean_r": r_val,
                            "mean_g": g_val,
                            "mean_b": b_val,
                        })

                    if meta:
                        polygon_result['meta'] = meta

                    results.append(polygon_result)

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

        image_path = self.get_local_path(
            image_url,
            ls_access_token=access_token,
            ls_host=hostname,
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
        point_labels = np.array(point_labels, dtype=np.int32) if point_labels else None
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

        # Preannotation path (no interactive context yet)
        has_interactive_context = isinstance(context, dict) and (
            'result' in context or 'prompts' in context or 'prompt' in context
        )
        if not has_interactive_context:
            # Resolve full config for preannotation path
            cfg = self._resolve_config(**kwargs)
            if not cfg['preannotate']:
                return self._empty_prediction_response()

            img_url = tasks[0]['data'][value]
            access_token = None
            hostname = None
            try:
                from org_api_middleware_v3 import get_credentials_for_task as _get_creds
                hostname, access_token, _ = _get_creds(tasks[0])
            except Exception:
                pass

            local_img_path = self.get_local_path(
                img_url,
                ls_access_token=access_token,
                ls_host=hostname,
                task_id=tasks[0].get('id')
            )
            image = cv2.imread(local_img_path)
            if image is None:
                return self._empty_prediction_response()
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
                return self._empty_prediction_response()

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

        default_label = self._get_default_label(from_name)
        prompt_payload = self._normalize_prompt_payload(context, default_label)
        if prompt_payload['empty']:
            return self._empty_prediction_response()

        image_width = prompt_payload['image_width']
        image_height = prompt_payload['image_height']
        point_coords = prompt_payload['point_coords']
        point_labels = prompt_payload['point_labels']
        input_box = prompt_payload['input_box']
        selected_label = prompt_payload['selected_label']

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
            local_img_path = self.get_local_path(
                tasks[0]['data'][value],
                ls_access_token=access_token,
                ls_host=hostname,
                task_id=tasks[0].get('id'),
            )
        except Exception:
            try:
                local_img_path = self.get_local_path(tasks[0]['data'][value], task_id=tasks[0].get('id'))
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
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3:
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
            x_coords = [pt[0] for pt in pixel_coords]
            y_coords = [pt[1] for pt in pixel_coords]
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

    def _compute_mask_geometry(self, mask: np.ndarray):
        """
        Compute simple geometry stats for a binary mask in **pixel** units.

        Returns a dict suitable for the `meta` field on Label Studio results:

        {
            "area": <int>,                # number of pixels inside the mask
            "bbox": {
                "x": <int>,              # left (min x) pixel
                "y": <int>,              # top (min y) pixel
                "width": <int>,          # width in pixels
                "height": <int>,         # height in pixels
            }
        }
        or None if the mask is empty.
        """
        if mask is None:
            return None

        mask_bool = (mask.astype(np.uint8) > 0)
        if not mask_bool.any():
            return None

        ys, xs = np.nonzero(mask_bool)
        if xs.size == 0 or ys.size == 0:
            return None

        x_min = int(xs.min())
        x_max = int(xs.max()) + 1
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1

        width_px = x_max - x_min
        height_px = y_max - y_min
        area_px = int(mask_bool.sum())

        return {
            "area": area_px,
            "bbox": {
                "x": x_min,
                "y": y_min,
                "width": width_px,
                "height": height_px,
            },
        }
