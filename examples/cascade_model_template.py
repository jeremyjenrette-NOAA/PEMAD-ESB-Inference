"""STRUCTURAL TEMPLATE -- not tested against real weights or the real
optics-models-hello-world `model.py` contract. See docs/two-stage-cascade.md
before using this.

Shows the shape of a two-stage cancer_crab -> species cascade to drop
into a forked `optics-models-hello-world` repo's `model.py`: run the
detector over a full image, crop each detected box, run the taxonomic
classifier on each crop, and write one combined KWCOCO-style output.

Confirm against the real template before relying on this:
  - the actual function signature `model.py` is expected to expose
    (this assumes `run_inference(input_dir, config, output_file_path)`
    per the hello-world README's description -- verify the exact name)
  - how crabdata_tax.pt should actually be loaded (this assumes
    `torch.load(...)` of a ready-to-use model object; adjust if it's a
    state_dict needing a model class, or a TorchScript export)
  - the classifier's real input size/normalization and output shape
    (this assumes a single flat species softmax; adjust if the
    hierarchical model has multiple heads, e.g. genus then species)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from ultralytics import YOLO

DEFAULT_DETECTOR_WEIGHTS = "best.pt"  # cancer_crab detector
DEFAULT_CLASSIFIER_WEIGHTS = "crabdata_tax.pt"  # taxonomic classifier

# Fill in from your training label map (index -> species/taxon label).
TAXONOMY_LABELS = [
    # "cancer_irroratus",
    # "cancer_borealis",
    # ...
]

_detector = None
_classifier = None


def _load_models(config: dict) -> None:
    global _detector, _classifier
    if _detector is None:
        _detector = YOLO(config.get("detector_weights", DEFAULT_DETECTOR_WEIGHTS))
    if _classifier is None:
        # TODO: confirm the real load path for crabdata_tax.pt.
        _classifier = torch.load(config.get("classifier_weights", DEFAULT_CLASSIFIER_WEIGHTS))
        _classifier.eval()


def _preprocess_for_classifier(crop: Image.Image) -> torch.Tensor:
    # TODO: match crabdata_tax.pt's real expected input size/normalization.
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(crop)


def _classify_crop(crop: Image.Image) -> Tuple[str, float]:
    tensor = _preprocess_for_classifier(crop).unsqueeze(0)
    with torch.no_grad():
        logits = _classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
    label = TAXONOMY_LABELS[pred_idx] if TAXONOMY_LABELS else str(pred_idx)
    return label, confidence


def _register_image(kwcoco_out: dict, image_path: Path, size: Tuple[int, int]) -> int:
    image_id = len(kwcoco_out["images"]) + 1
    kwcoco_out["images"].append(
        {"id": image_id, "file_name": image_path.name, "width": size[0], "height": size[1]}
    )
    return image_id


def run_inference(input_dir: str, config: dict, output_file_path: str) -> None:
    """Entry point matching the hello-world model.py contract (per its
    README: read input_dir, use config for weights/thresholds, write to
    output_file_path). Confirm the exact expected signature/name against
    the real template.
    """
    _load_models(config)
    detector_conf = config.get("detector_conf", 0.25)

    kwcoco_out: dict = {"images": [], "annotations": [], "categories": []}
    ann_id = 1

    image_paths = sorted(p for p in Path(input_dir).iterdir() if p.is_file())
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_id = _register_image(kwcoco_out, image_path, image.size)

        # Stage 1: detect cancer_crab candidates.
        results = _detector.predict(image, conf=detector_conf, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            crop = image.crop((x1, y1, x2, y2))

            # Stage 2: classify the crop's species.
            species_label, species_conf = _classify_crop(crop)

            kwcoco_out["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_name": species_label,
                    "detector_confidence": float(box.conf[0]),
                    "classifier_confidence": species_conf,
                }
            )
            ann_id += 1

    import json

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(kwcoco_out, f, indent=2)
