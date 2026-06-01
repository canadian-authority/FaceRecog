from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from facerec_core import (
    compute_face_encoding,
    detect_faces,
    draw_face_label,
    iter_image_paths,
    load_known_encodings,
    load_models,
    match_face,
    rect_to_bounds,
)


PROJECT_DIR = Path(__file__).resolve().parent


def default_path(filename: str) -> Path:
    candidate = PROJECT_DIR / filename
    if candidate.exists():
        return candidate
    return Path(filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize known faces in every image under a folder.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_path("train_faces"),
        help="Folder of images to scan recursively.",
    )
    parser.add_argument(
        "--encodings",
        type=Path,
        default=default_path("encodings.pickle"),
        help="Pickle file containing known face encodings.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=default_path("."),
        help="Folder containing the dlib .dat model files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recognized_output"),
        help="Folder where annotated images will be written.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("recognition_results.csv"),
        help="CSV file for recognition results.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Maximum Euclidean distance for a face match. Lower is stricter.",
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
        default=0,
        help="Detection upsampling. Increase to 1 for small faces.",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=1,
        help="Number of descriptor jitters. Increase for accuracy, decrease for speed.",
    )
    parser.add_argument(
        "--unknown-name",
        default="Unknown",
        help="Label to use when the closest match is above the threshold.",
    )
    return parser.parse_args()


def annotated_output_path(input_root: Path, image_path: Path, output_dir: Path) -> Path:
    relative = image_path.relative_to(input_root)
    return output_dir / relative.with_suffix(".jpg")


def write_results(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path = csv_path.expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "face_index",
        "name",
        "matched",
        "distance",
        "left",
        "top",
        "right",
        "bottom",
        "status",
        "output_image",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    print(f"[INFO] Loading encodings from {args.encodings}")
    known_encodings, known_names = load_known_encodings(args.encodings)
    print(f"[INFO] Loaded {len(known_encodings)} known face encodings.")

    print(f"[INFO] Loading dlib models from {args.models_dir}")
    models = load_models(args.models_dir, args.detector)
    print(f"[INFO] Using {models.detector_kind.upper()} detector.")

    image_paths = list(iter_image_paths(input_dir))
    if not image_paths:
        raise SystemExit(f"No images found under: {input_dir}")

    rows: list[dict[str, object]] = []
    total_faces = 0
    matched_faces = 0
    unknown_faces = 0
    no_face_images = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path}")
        image_bgr = cv2.imread(str(image_path))
        output_path = annotated_output_path(input_dir, image_path, output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if image_bgr is None:
            rows.append(
                {
                    "image": str(image_path),
                    "face_index": "",
                    "name": "",
                    "matched": False,
                    "distance": "",
                    "left": "",
                    "top": "",
                    "right": "",
                    "bottom": "",
                    "status": "read_error",
                    "output_image": "",
                }
            )
            continue

        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rects = detect_faces(models, rgb_image, args.upsample)

        if not rects:
            no_face_images += 1
            rows.append(
                {
                    "image": str(image_path),
                    "face_index": "",
                    "name": "",
                    "matched": False,
                    "distance": "",
                    "left": "",
                    "top": "",
                    "right": "",
                    "bottom": "",
                    "status": "no_face",
                    "output_image": str(output_path),
                }
            )
        else:
            for face_index, rect in enumerate(rects, start=1):
                total_faces += 1
                encoding = compute_face_encoding(models, rgb_image, rect, args.jitter)
                name, distance, matched = match_face(
                    known_encodings,
                    known_names,
                    encoding,
                    args.threshold,
                    args.unknown_name,
                )
                if matched:
                    matched_faces += 1
                else:
                    unknown_faces += 1

                left, top, right, bottom = rect_to_bounds(rect, image_bgr.shape)
                label = f"{name} ({distance:.3f})" if distance != float("inf") else name
                draw_face_label(image_bgr, rect, label, matched)

                rows.append(
                    {
                        "image": str(image_path),
                        "face_index": face_index,
                        "name": name,
                        "matched": matched,
                        "distance": f"{distance:.6f}" if distance != float("inf") else "",
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "status": "matched" if matched else "unknown",
                        "output_image": str(output_path),
                    }
                )

        cv2.imwrite(str(output_path), image_bgr)

    write_results(args.csv, rows)

    print("[INFO] Done.")
    print(f"[INFO] Images scanned: {len(image_paths)}")
    print(f"[INFO] Faces found: {total_faces}")
    print(f"[INFO] Matched faces: {matched_faces}")
    print(f"[INFO] Unknown faces: {unknown_faces}")
    print(f"[INFO] Images with no face: {no_face_images}")
    print(f"[INFO] CSV: {args.csv.expanduser().resolve()}")
    print(f"[INFO] Annotated images: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


