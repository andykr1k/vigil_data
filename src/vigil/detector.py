"""Person detection with a transformers object detector (RT-DETR by default)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from .config import DetectorConfig


class PersonDetector:
    def __init__(self, cfg: DetectorConfig, device: str):
        self.cfg = cfg
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(cfg.model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(cfg.model_id).to(device).eval()
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.person_id = label2id["person"]

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.detect_many([rgb])[0]

    @torch.inference_mode()
    def detect_many(self, images: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per image: (N, 4) xyxy pixel boxes and (N,) scores for people, highest score first.

        All images go through the model as one batch (one camera image each).
        """
        if not images:
            return []
        inputs = self.processor(images=[Image.fromarray(im) for im in images],
                                return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=[im.shape[:2] for im in images],
            threshold=self.cfg.score_threshold,
        )
        out = []
        for res in results:
            keep = res["labels"] == self.person_id
            boxes = res["boxes"][keep].float().cpu().numpy()
            scores = res["scores"][keep].float().cpu().numpy()
            order = np.argsort(-scores)
            out.append((boxes[order], scores[order]))
        return out
