from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import dlib
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CNN_FACE_DETECTOR = "mmod_human_face_detector.dat"
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL = "dlib_face_recognition_resnet_model_v1.dat"


@dataclass(frozen=True)
class FaceModels:
    detector: object
    detector_kind: str
    predictor: dlib.shape_predictor
    recognizer: dlib.face_recognition_model_v1


def iter_image_paths(folder: Path) -> Iterable[Path]:
    """Yield image files below a folder in stable order."""
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_models(models_dir: Path, detector_kind: str = "auto") -> FaceModels:
    models_dir = models_dir.expanduser().resolve()
    predictor_path = require_file(models_dir / SHAPE_PREDICTOR, "shape predictor")
    recognizer_path = require_file(models_dir / FACE_RECOGNITION_MODEL, "face recognition model")
    cnn_path = models_dir / CNN_FACE_DETECTOR

    detector_kind = detector_kind.lower()
    if detector_kind not in {"auto", "cnn", "hog"}:
        raise ValueError("detector_kind must be one of: auto, cnn, hog")

    if detector_kind == "auto":
        detector_kind = "cnn" if cnn_path.is_file() else "hog"

    if detector_kind == "cnn":
        require_file(cnn_path, "CNN face detector")
        detector = dlib.cnn_face_detection_model_v1(str(cnn_path))
    else:
        detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor(str(predictor_path))
    recognizer = dlib.face_recognition_model_v1(str(recognizer_path))

    return FaceModels(
        detector=detector,
        detector_kind=detector_kind,
        predictor=predictor,
        recognizer=recognizer,
    )


def detect_faces(models: FaceModels, rgb_image: np.ndarray, upsample: int = 0) -> list[dlib.rectangle]:
    results = models.detector(rgb_image, upsample)
    if models.detector_kind == "cnn":
        return [result.rect for result in results]
    return list(results)


def compute_face_encoding(
    models: FaceModels,
    rgb_image: np.ndarray,
    rect: dlib.rectangle,
    num_jitters: int = 1,
) -> np.ndarray:
    shape = models.predictor(rgb_image, rect)
    descriptor = models.recognizer.compute_face_descriptor(rgb_image, shape, num_jitters)
    return np.asarray(descriptor, dtype=np.float32)


def load_known_encodings(encodings_path: Path) -> tuple[np.ndarray, list[str]]:
    encodings_path = require_file(encodings_path.expanduser().resolve(), "encodings pickle")
    with encodings_path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict) or "encodings" not in data or "names" not in data:
        raise ValueError("Encodings file must contain a dict with 'encodings' and 'names'.")

    encodings = np.asarray(data["encodings"], dtype=np.float32)
    if encodings.size == 0:
        encodings = np.empty((0, 128), dtype=np.float32)
    elif encodings.ndim != 2:
        encodings = encodings.reshape((len(data["encodings"]), -1))

    names = [str(name) for name in data["names"]]
    if len(encodings) != len(names):
        raise ValueError("Encodings and names counts do not match.")

    return encodings, names


def save_known_encodings(encodings_path: Path, encodings: list[np.ndarray], names: list[str]) -> None:
    encodings_path = encodings_path.expanduser().resolve()
    encodings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"encodings": encodings, "names": names}
    with encodings_path.open("wb") as handle:
        pickle.dump(payload, handle)


def match_face(
    known_encodings: np.ndarray,
    known_names: list[str],
    face_encoding: np.ndarray,
    threshold: float,
    unknown_name: str = "Unknown",
) -> tuple[str, float, bool]:
    if known_encodings.size == 0:
        return unknown_name, float("inf"), False

    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    matched = best_distance <= threshold
    name = known_names[best_index] if matched else unknown_name
    return name, best_distance, matched


def rect_to_bounds(rect: dlib.rectangle, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    left = max(0, min(width - 1, int(rect.left())))
    top = max(0, min(height - 1, int(rect.top())))
    right = max(0, min(width - 1, int(rect.right())))
    bottom = max(0, min(height - 1, int(rect.bottom())))
    return left, top, right, bottom


def draw_face_label(
    image_bgr: np.ndarray,
    rect: dlib.rectangle,
    label: str,
    matched: bool,
) -> None:
    left, top, right, bottom = rect_to_bounds(rect, image_bgr.shape)
    color = (0, 180, 0) if matched else (0, 0, 220)

    cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_top = max(0, top - text_height - baseline - 8)
    label_right = min(image_bgr.shape[1] - 1, left + text_width + 12)

    cv2.rectangle(image_bgr, (left, label_top), (label_right, top), color, cv2.FILLED)
    cv2.putText(
        image_bgr,
        label,
        (left + 6, max(text_height + 2, top - baseline - 4)),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )
