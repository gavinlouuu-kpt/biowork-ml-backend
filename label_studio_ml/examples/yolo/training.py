import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import requests
import yaml
from ultralytics import YOLO

from control_models.base import MODEL_ROOT, _model_cache

logger = logging.getLogger(__name__)


@dataclass
class ControlSpec:
    name: str
    control_type: str
    labels: List[str]
    data_key: str


class YoloAutoTrainer:
    def __init__(self, backend):
        self.backend = backend
        self.project_id = str(backend.project_id)
        self.project_root = Path(
            os.getenv("YOLO_TRAIN_PROJECT_ROOT", "/data/server/yolo_autotrain")
        ) / f"project_{self.project_id}"
        self.project_root.mkdir(parents=True, exist_ok=True)

        self.epochs = int(os.getenv("YOLO_TRAIN_EPOCHS", "25"))
        self.imgsz = int(os.getenv("YOLO_TRAIN_IMGSZ", "1024"))
        self.batch = int(os.getenv("YOLO_TRAIN_BATCH", "8"))
        self.workers = int(os.getenv("YOLO_TRAIN_WORKERS", "2"))
        self.patience = int(os.getenv("YOLO_TRAIN_PATIENCE", "20"))

        self.base_detect_model = os.getenv("YOLO_TRAIN_BASE_MODEL_DETECT", "yolov8m.pt")
        self.base_segment_model = os.getenv("YOLO_TRAIN_BASE_MODEL_SEGMENT", "yolov8n-seg.pt")

    def run(self, event: str, data: Optional[Dict]) -> Dict:
        tasks = self._extract_annotated_tasks(data or {})
        if not tasks:
            return {
                "status": "skipped",
                "reason": "No annotated tasks found for training",
                "project_id": self.project_id,
            }

        controls = self._parse_image_controls(self.backend.label_config)
        if not controls:
            return {
                "status": "skipped",
                "reason": "Label config has no Image RectangleLabels/PolygonLabels controls",
                "project_id": self.project_id,
            }

        dataset = self._build_dataset(tasks, controls)
        if dataset["num_samples"] == 0:
            return {
                "status": "skipped",
                "reason": "No supported regions found in annotations (RectangleLabels/PolygonLabels)",
                "project_id": self.project_id,
            }

        train_result = self._train(dataset)
        self._activate_model(train_result["best_model_path"], train_result["model_version"])

        return {
            "status": "trained",
            "event": event,
            "project_id": self.project_id,
            "model_version": train_result["model_version"],
            "active_model_path": train_result["active_model_path"],
            "task": dataset["task"],
            "num_samples": dataset["num_samples"],
            "num_train": dataset["num_train"],
            "num_val": dataset["num_val"],
            "dataset_path": str(dataset["dataset_dir"]),
        }

    def _extract_annotated_tasks(self, data: Dict) -> List[Dict]:
        tasks = data.get("annotations")
        if tasks:
            return self._normalize_task_annotations(tasks)

        # Webhook payload often sends only current task + annotation.
        if isinstance(data.get("task"), dict) and isinstance(data.get("annotation"), dict):
            task = dict(data["task"])
            task.setdefault("annotations", [])
            task["annotations"].append(data["annotation"])
            return [task]

        # Webhook START_TRAINING payload may only contain project metadata.
        return self._fetch_annotated_tasks_from_label_studio()

    def _normalize_task_annotations(self, tasks: List[Dict]) -> List[Dict]:
        normalized: List[Dict] = []
        for task in tasks:
            task_copy = dict(task)
            annotations = task_copy.get("annotations") or []
            if isinstance(annotations, str):
                try:
                    annotations = json.loads(annotations)
                except json.JSONDecodeError:
                    annotations = []
            task_copy["annotations"] = annotations
            normalized.append(task_copy)
        return normalized

    def _resolve_credentials(self) -> Tuple[str, str]:
        """Resolve Label Studio host and API token via middleware or env vars."""
        host = ""
        token = ""

        # Try the organization middleware first (auto-resolves tokens from Label Studio DB)
        if os.getenv("USE_ORG_MIDDLEWARE", "").lower() in ("true", "1", "yes"):
            try:
                from org_api_middleware_v3 import get_middleware

                middleware = get_middleware()
                project_id_int = int(self.project_id)
                host, token, token_type = middleware.get_credentials_for_project(project_id_int)
                if host and token:
                    logger.info(
                        "Middleware resolved credentials for project %s (token_type=%s)",
                        self.project_id, token_type,
                    )
            except Exception as exc:
                logger.debug(
                    "Middleware credential resolution failed for project %s: %s",
                    self.project_id, exc,
                )

        # Fall back to backend storage / env vars
        if not host:
            host = (self.backend.get("ls_host") or os.getenv("LABEL_STUDIO_HOST") or "").strip()
        if not token:
            token = (self.backend.get("ls_access_token") or os.getenv("LABEL_STUDIO_API_KEY") or "").strip()

        return host, token

    def _fetch_annotated_tasks_from_label_studio(self) -> List[Dict]:
        host, token = self._resolve_credentials()

        if not host or not token:
            logger.warning(
                "Cannot fetch annotated tasks: missing Label Studio host/token for project %s",
                self.project_id,
            )
            return []

        host = host.rstrip("/")
        url = f"{host}/api/tasks/"
        params = {
            "project": self.project_id,
            "fields": "all",
            "only_annotated": "true",
            "page_size": 100,
            "page": 1,
        }

        headers_candidates = [
            {"Authorization": f"Token {token}"},
            {"Authorization": f"Bearer {token}"},
        ]

        for headers in headers_candidates:
            try:
                all_tasks: List[Dict] = []
                next_url = url
                next_params = dict(params)
                while next_url:
                    response = requests.get(next_url, params=next_params, headers=headers, timeout=30)
                    if response.status_code in (401, 403):
                        all_tasks = []
                        break
                    response.raise_for_status()
                    payload = response.json()
                    tasks = payload.get("tasks", [])
                    all_tasks.extend(tasks)
                    next_url = payload.get("next")
                    next_params = None if next_url else {}
                if all_tasks:
                    return self._normalize_task_annotations(all_tasks)
            except Exception as exc:
                logger.warning("Failed to fetch tasks for training from Label Studio: %s", exc)
        return []

    def _parse_image_controls(self, label_config: str) -> Dict[str, ControlSpec]:
        root = ET.fromstring(label_config)
        object_keys: Dict[str, str] = {}
        controls: Dict[str, ControlSpec] = {}

        # Build object name -> task data key map.
        for el in root.iter():
            if el.tag != "Image":
                continue
            name = (el.attrib.get("name") or "").strip()
            value = (el.attrib.get("value") or "").strip()
            if not name or not value.startswith("$"):
                continue
            object_keys[name] = value[1:]

        for el in root.iter():
            if el.tag not in ("RectangleLabels", "PolygonLabels"):
                continue
            control_name = (el.attrib.get("name") or "").strip()
            to_name = (el.attrib.get("toName") or "").strip()
            if not control_name or not to_name:
                continue
            data_key = object_keys.get(to_name)
            if not data_key:
                continue
            labels = []
            for child in el:
                if child.tag != "Label":
                    continue
                label = (child.attrib.get("value") or "").strip()
                if label:
                    labels.append(label)
            if labels:
                controls[control_name] = ControlSpec(
                    name=control_name,
                    control_type=el.tag,
                    labels=labels,
                    data_key=data_key,
                )
        return controls

    def _build_dataset(self, tasks: List[Dict], controls: Dict[str, ControlSpec]) -> Dict:
        run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dataset_dir = self.project_root / "datasets" / run_stamp
        images_train = dataset_dir / "images" / "train"
        images_val = dataset_dir / "images" / "val"
        labels_train = dataset_dir / "labels" / "train"
        labels_val = dataset_dir / "labels" / "val"
        for path in (images_train, images_val, labels_train, labels_val):
            path.mkdir(parents=True, exist_ok=True)

        class_names = self._collect_class_names(controls)
        class_map = {label: idx for idx, label in enumerate(class_names)}

        # Build samples first, then split.
        samples: List[Tuple[Path, List[str]]] = []
        has_polygon = False
        for task in tasks:
            label_lines, image_path, sample_has_polygon = self._build_sample(task, controls, class_map)
            if not image_path:
                continue
            has_polygon = has_polygon or sample_has_polygon
            if not label_lines:
                continue
            samples.append((image_path, label_lines))

        if not samples:
            return {
                "dataset_dir": dataset_dir,
                "num_samples": 0,
                "num_train": 0,
                "num_val": 0,
                "task": "detect",
            }

        # If any polygon exists, run a segmentation task. Convert rectangle lines to polygon lines.
        task_type = "segment" if has_polygon else "detect"
        if task_type == "segment":
            converted_samples: List[Tuple[Path, List[str]]] = []
            for image_path, lines in samples:
                converted = []
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        cls_id, xc, yc, w, h = parts
                        xc = float(xc)
                        yc = float(yc)
                        w = float(w)
                        h = float(h)
                        x1 = max(0.0, xc - w / 2.0)
                        y1 = max(0.0, yc - h / 2.0)
                        x2 = min(1.0, xc + w / 2.0)
                        y2 = min(1.0, yc + h / 2.0)
                        converted.append(
                            f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"
                        )
                    else:
                        converted.append(line)
                converted_samples.append((image_path, converted))
            samples = converted_samples

        num_samples = len(samples)
        num_val = 1 if num_samples <= 5 else max(1, int(num_samples * 0.2))
        if num_samples == 1:
            num_val = 1
        num_train = max(1, num_samples - num_val)

        for idx, (src_image, label_lines) in enumerate(samples):
            split = "train" if idx < num_train else "val"
            image_name = f"{idx:06d}_{self._safe_name(src_image.name)}"
            target_image = (images_train if split == "train" else images_val) / image_name
            target_label = (labels_train if split == "train" else labels_val) / (
                Path(image_name).stem + ".txt"
            )

            shutil.copy2(src_image, target_image)
            target_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        # When num_samples is very small (e.g. 1), the split logic above may leave the
        # validation directory empty. YOLO requires a validation set, so copy the train
        # images/labels into val to satisfy the requirement.
        val_images = list(images_val.iterdir())
        if not val_images:
            logger.warning(
                "Validation set is empty (%d sample(s) total). Copying train data to val.",
                num_samples,
            )
            for train_image in images_train.iterdir():
                if train_image.is_file():
                    val_image = images_val / train_image.name
                    shutil.copy2(train_image, val_image)
            for train_label in labels_train.iterdir():
                if train_label.is_file():
                    val_label = labels_val / train_label.name
                    shutil.copy2(train_label, val_label)

        data_yaml = dataset_dir / "data.yaml"
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train",
                "val": "images/val",
                "names": {idx: name for idx, name in enumerate(class_names)},
            },
            data_yaml.open("w", encoding="utf-8"),
            sort_keys=False,
        )

        return {
            "dataset_dir": dataset_dir,
            "data_yaml": data_yaml,
            "num_samples": num_samples,
            "num_train": num_train,
            "num_val": num_val,
            "task": task_type,
        }

    def _collect_class_names(self, controls: Dict[str, ControlSpec]) -> List[str]:
        names: List[str] = []
        for spec in controls.values():
            for label in spec.labels:
                if label not in names:
                    names.append(label)
        return names

    def _build_sample(
        self, task: Dict, controls: Dict[str, ControlSpec], class_map: Dict[str, int]
    ) -> Tuple[List[str], Optional[Path], bool]:
        annotations = task.get("annotations") or []
        if not annotations:
            return [], None, False

        chosen_annotation = self._choose_annotation(annotations)
        if not chosen_annotation:
            return [], None, False

        results = chosen_annotation.get("result") or []
        if not results:
            return [], None, False

        lines: List[str] = []
        has_polygon = False
        image_path: Optional[Path] = None

        for region in results:
            from_name = region.get("from_name")
            region_type = (region.get("type") or "").lower()
            value = region.get("value") or {}
            spec = controls.get(from_name)
            if not spec:
                continue

            if spec.control_type == "RectangleLabels" and region_type != "rectanglelabels":
                continue
            if spec.control_type == "PolygonLabels" and region_type != "polygonlabels":
                continue

            label_value_key = (
                "rectanglelabels" if spec.control_type == "RectangleLabels" else "polygonlabels"
            )
            labels = value.get(label_value_key) or []
            if not labels:
                continue
            label = labels[0]
            if label not in class_map:
                continue
            class_id = class_map[label]

            if image_path is None:
                image_url = (task.get("data") or {}).get(spec.data_key)
                if not image_url:
                    return [], None, False
                task_id = task.get("id")
                local = self.backend.get_local_path(
                    image_url,
                    task_id=task_id,
                    ls_host=self.backend.get("ls_host"),
                    ls_access_token=self.backend.get("ls_access_token"),
                )
                image_path = Path(local)

            if spec.control_type == "RectangleLabels":
                line = self._rectangle_to_yolo_line(class_id, value)
                if line:
                    lines.append(line)
            else:
                line = self._polygon_to_yolo_line(class_id, value)
                if line:
                    has_polygon = True
                    lines.append(line)

        return lines, image_path, has_polygon

    def _choose_annotation(self, annotations: List[Dict]) -> Optional[Dict]:
        valid = [a for a in annotations if not a.get("was_cancelled")]
        if not valid:
            return None
        ground_truth = [a for a in valid if a.get("ground_truth")]
        if ground_truth:
            return ground_truth[-1]
        return valid[-1]

    def _rectangle_to_yolo_line(self, class_id: int, value: Dict) -> Optional[str]:
        try:
            x = float(value["x"])
            y = float(value["y"])
            w = float(value["width"])
            h = float(value["height"])
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        xc = self._clip01((x + w / 2.0) / 100.0)
        yc = self._clip01((y + h / 2.0) / 100.0)
        wn = self._clip01(w / 100.0)
        hn = self._clip01(h / 100.0)
        return f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"

    def _polygon_to_yolo_line(self, class_id: int, value: Dict) -> Optional[str]:
        points = value.get("points") or []
        if len(points) < 3:
            return None
        flat: List[str] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            x = self._clip01(float(point[0]) / 100.0)
            y = self._clip01(float(point[1]) / 100.0)
            flat.extend([f"{x:.6f}", f"{y:.6f}"])
        if len(flat) < 6:
            return None
        return f"{class_id} " + " ".join(flat)

    def _train(self, dataset: Dict) -> Dict:
        task = dataset["task"]
        base_model = self.base_segment_model if task == "segment" else self.base_detect_model
        source_model = self._resolve_source_model(base_model)

        model = YOLO(source_model)
        train_name = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        model.train(
            data=str(dataset["data_yaml"]),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            workers=self.workers,
            patience=self.patience,
            project=str(self.project_root / "runs"),
            name=train_name,
            exist_ok=False,
        )

        trainer = getattr(model, "trainer", None)
        best_model_path = None
        if trainer is not None:
            best_model_path = getattr(trainer, "best", None)
        if not best_model_path:
            fallback = self.project_root / "runs" / train_name / "weights" / "best.pt"
            best_model_path = str(fallback)

        best_path = Path(best_model_path)
        if not best_path.exists():
            raise FileNotFoundError(f"Trained checkpoint not found: {best_path}")

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        relative_target = Path("autotrain") / f"project_{self.project_id}" / "best.pt"
        absolute_target = Path(MODEL_ROOT) / relative_target
        absolute_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, absolute_target)

        model_version = f"yolo-auto-{task}-{timestamp}"
        return {
            "best_model_path": str(absolute_target),
            "active_model_path": str(relative_target).replace("\\", "/"),
            "model_version": model_version,
        }

    def _resolve_source_model(self, fallback_model: str) -> str:
        # Prefer the active trained model if present.
        active_path = (self.backend.get("active_model_path") or "").strip()
        if active_path:
            absolute_active = Path(MODEL_ROOT) / active_path
            if absolute_active.exists():
                return str(absolute_active)

        # Then try fallback path in MODEL_ROOT.
        in_model_root = Path(MODEL_ROOT) / fallback_model
        if in_model_root.exists():
            return str(in_model_root)
        return fallback_model

    def _activate_model(self, absolute_model_path: str, model_version: str) -> None:
        absolute_path = Path(absolute_model_path)
        rel_path = absolute_path.relative_to(Path(MODEL_ROOT))
        rel_str = str(rel_path).replace("\\", "/")

        self.backend.set("active_model_path", rel_str)
        self.backend.set("model_version", model_version)

        # Keep inference cache consistent: next predict call should reload new checkpoint.
        _model_cache.clear()

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", name)
