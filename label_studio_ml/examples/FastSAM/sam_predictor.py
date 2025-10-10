import os
import logging
import time
import pathlib
import numpy as np

from typing import List, Dict, Optional, Tuple
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path

logger = logging.getLogger(__name__)
_MODELS_DIR = pathlib.Path(__file__).parent / "models"

FASTSAM_CHECKPOINT = os.environ.get("FASTSAM_CHECKPOINT", _MODELS_DIR / "FastSAM-x.pt")
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")

# Organization middleware support (V3)
USE_ORG_MIDDLEWARE = os.getenv('USE_ORG_MIDDLEWARE', 'false').lower() in ('true', '1', 'yes')
_middleware_instance = None

if USE_ORG_MIDDLEWARE:
    try:
        from org_api_middleware_v3 import get_middleware
        _middleware_instance = get_middleware()
        logger.info("Organization middleware enabled for FastSAM predictor (V3)")
    except ImportError:
        logger.warning("Organization middleware requested but not available, using static tokens")
        USE_ORG_MIDDLEWARE = False


def get_credentials_for_task(task: Optional[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve Label Studio host and access token for a task using middleware if enabled."""
    if USE_ORG_MIDDLEWARE and _middleware_instance and task:
        try:
            if 'project' in task:
                project_id = task['project']
                if isinstance(project_id, str):
                    host, token, token_type = _middleware_instance.get_credentials_from_project_uid(project_id)
                else:
                    host, token, token_type = _middleware_instance.get_credentials_for_project(project_id)
                if host and token:
                    logger.debug(f"Using {token_type} token from middleware for project {project_id}")
                    return host, token
                logger.warning(f"Middleware failed to get credentials for project {project_id}, using fallback")
        except Exception as e:
            logger.error(f"Error getting credentials from middleware: {e}")
    return LABEL_STUDIO_HOST, LABEL_STUDIO_ACCESS_TOKEN


class SAMPredictor:
    """FastSAM-only predictor wrapper used by the dedicated server."""

    def __init__(self, model_choice: str = 'FastSAM'):
        if model_choice != 'FastSAM':
            logger.warning(f"Dedicated FastSAM server received model_choice={model_choice}; forcing FastSAM")
        self.model_choice = 'FastSAM'
        self.model_checkpoint = str(FASTSAM_CHECKPOINT)
        if not self.model_checkpoint:
            raise FileNotFoundError("FASTSAM_CHECKPOINT is not set: please set it to the path to the FastSAM checkpoint")

        # device selection
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"
        logger.info(f"FastSAM predictor device: {self.device}")

        # Lazy FastSAM init
        self._fastsam_model = None

    @property
    def model_name(self) -> str:
        return f"{self.model_choice}:{self.model_checkpoint}:{self.device}"

    def predict(
        self,
        img_path: str,
        point_coords: Optional[List[List]] = None,
        point_labels: Optional[List] = None,
        input_box: Optional[List] = None,
        task: Optional[Dict] = None
    ) -> Dict:
        """Run FastSAM prediction with optional point or box prompts, else everything-mode."""
        predict_start_time = time.time()
        logger.debug(f"Starting FastSAM prediction for {img_path}")

        # Resolve image path with credentials
        hostname, access_token = get_credentials_for_task(task)
        image_path = get_local_path(
            img_path,
            access_token=access_token,
            hostname=hostname,
            task_id=task.get('id') if task else None
        )

        # Import FastSAM components
        try:
            from ultralytics import FastSAM
            from fastsam_prompt import FastSAMPrompt
        except Exception as e:
            logger.error(f"Failed to import FastSAM components: {e}")
            raise

        # Initialize model once
        if self._fastsam_model is None:
            logger.info(f"Initializing FastSAM model with checkpoint: {self.model_checkpoint}")
            self._fastsam_model = FastSAM(self.model_checkpoint)

        # Everything inference (FastSAM produces candidate masks)
        inference_start_time = time.time()
        everything_results = self._fastsam_model(
            source=image_path,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9,
        )
        logger.debug(f"FastSAM base inference took {time.time() - inference_start_time:.4f}s")

        # Prompting
        prompt_process = FastSAMPrompt(image_path, everything_results, device=self.device)
        if point_coords and point_labels:
            ann = prompt_process.point_prompt(point_coords, point_labels)
        elif input_box:
            ann = prompt_process.box_prompt(input_box)
        else:
            ann = prompt_process.everything_prompt()

        masks: List[np.ndarray] = []
        probs: List[float] = []
        if ann is not None and len(ann) > 0:
            for mask in ann:
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                mask = (mask > 0).astype(np.uint8)
                masks.append(mask)
                probs.append(0.8)

        if not masks:
            logger.warning("FastSAM: No masks generated")
            return {"masks": [], "probs": []}

        logger.debug(f"FastSAM prediction completed in {time.time() - predict_start_time:.4f}s")
        return {"masks": masks, "probs": probs}


