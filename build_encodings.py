from __future__ import annotations

import argparse
import gc
from pathlib import Path

import cv2

from facerec_core import (
    compute_face_encoding,
    detect_faces,
    iter_image_paths,
    load_models,
    save_known_encodings,
)


PROJECT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = Path(r"C:\Users\daule\Desktop\MLAI")


def default_path(filename: str) -> Path:
    for base_dir in (PROJECT_DIR, DRAFT_DIR):
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    return Path(filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build encodings.pickle from a labeled folder of face images.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_path("facesm6"),
        help="Folder with one subfolder per person, e.g. faces/Person/image.jpg.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("encodings.pickle"),
        help="Where to write the generated pickle.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=default_path("."),
        help="Folder containing the dlib .dat model files.",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "cnn", "hog"),
        default="auto",
        help="Face detector to use. CNN is more accurate; HOG is faster on CPU.",
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=1,
        help="Detection upsampling. Increase for small faces.",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=10,
        help="Number of descriptor jitters. Higher is slower but can improve stability.",
    )
    parser.add_argument(
        "--allow-multiple",
        action="store_true",
        help="Encode every face in an image. By default, images with multiple faces are skipped.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.expanduser().resolve()

    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    print(f"[INFO] Loading dlib models from {args.models_dir}")
    models = load_models(args.models_dir, args.detector)
    print(f"[INFO] Using {models.detector_kind.upper()} detector.")

    image_paths = list(iter_image_paths(dataset_dir))
    if not image_paths:
        raise SystemExit(f"No images found under: {dataset_dir}")

    encodings = []
    names = []
    skipped_no_face = 0
    skipped_multiple = 0
    read_errors = 0

    for index, image_path in enumerate(image_paths, start=1):
        if args.progress_every and (index == 1 or index % args.progress_every == 0):
            print(f"[{index}/{len(image_paths)}] {image_path}")

        person_name = image_path.parent.name
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            read_errors += 1
            continue

        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rects = detect_faces(models, rgb_image, args.upsample)

        if not rects:
            skipped_no_face += 1
            continue

        if len(rects) > 1 and not args.allow_multiple:
            skipped_multiple += 1
            print(f"[WARN] Multiple faces; skipped to avoid label pollution: {image_path}")
            continue

        for rect in rects:
            encodings.append(compute_face_encoding(models, rgb_image, rect, args.jitter))
            names.append(person_name)

        del image_bgr, rgb_image, rects
        gc.collect()

    if not encodings:
        raise SystemExit("No face encodings were generated.")

    save_known_encodings(args.output, encodings, names)
    print("[INFO] Done.")
    print(f"[INFO] Encodings written: {len(encodings)}")
    print(f"[INFO] People found: {len(set(names))}")
    print(f"[INFO] Output: {args.output.expanduser().resolve()}")
    print(f"[INFO] Skipped no-face images: {skipped_no_face}")
    print(f"[INFO] Skipped multiple-face images: {skipped_multiple}")
    print(f"[INFO] Read errors: {read_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

