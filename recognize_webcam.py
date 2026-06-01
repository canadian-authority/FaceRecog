from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import dlib

from facerec_core import (
    compute_face_encoding,
    detect_faces,
    draw_face_label,
    load_known_encodings,
    load_models,
    match_face,
)


PROJECT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = Path(r"C:\Users\daule\Desktop\MLAI")


BACKENDS = {
    "auto": None,
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def default_path(filename: str) -> Path:
    for base_dir in (PROJECT_DIR, DRAFT_DIR):
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    return Path(filename)


def parse_camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize known faces from a webcam stream.",
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index, video file, or stream URL. Default: 0.",
    )
    parser.add_argument(
        "--backend",
        choices=tuple(BACKENDS),
        default="dshow",
        help="OpenCV camera backend. On Windows, dshow often opens webcams fastest.",
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
        "--detector",
        choices=("auto", "cnn", "hog"),
        default="auto",
        help="Face detector to use. CNN is best with your CUDA-enabled dlib.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Maximum Euclidean distance for a face match. Lower is stricter.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=640,
        help="Resize camera frames to this width before recognition. Use 0 for original size.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=2,
        help="Run recognition every N frames and reuse labels between processed frames.",
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=0,
        help="Detection upsampling. Increase to 1 for small/distant faces.",
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
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not mirror the webcam preview.",
    )
    parser.add_argument(
        "--window-name",
        default="Face Recognition Webcam",
        help="OpenCV display window title.",
    )
    return parser.parse_args()


def open_capture(camera: str, backend_name: str) -> cv2.VideoCapture:
    source = parse_camera_source(camera)
    backend = BACKENDS[backend_name]
    if backend is None:
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source, backend)


def resize_for_recognition(frame, frame_width: int):
    if frame_width <= 0:
        return frame

    height, width = frame.shape[:2]
    if width == frame_width:
        return frame

    ratio = frame_width / float(width)
    new_height = int(height * ratio)
    return cv2.resize(frame, (frame_width, new_height))


def draw_status(frame, fps: float, detector_kind: str) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | {detector_kind.upper()} | q/Esc to quit",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )


def main() -> int:
    args = parse_args()
    process_every = max(1, args.process_every)

    print(f"[INFO] Loading encodings from {args.encodings}")
    known_encodings, known_names = load_known_encodings(args.encodings)
    print(f"[INFO] Loaded {len(known_encodings)} known face encodings.")

    print(f"[INFO] Loading dlib models from {args.models_dir}")
    models = load_models(args.models_dir, args.detector)
    print(f"[INFO] Using {models.detector_kind.upper()} detector.")
    print(f"[INFO] dlib CUDA enabled: {dlib.DLIB_USE_CUDA}")

    capture = open_capture(args.camera, args.backend)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera/source: {args.camera}")

    frame_count = 0
    face_results: list[tuple[object, str, float, bool]] = []
    fps = 0.0
    fps_start = time.time()
    fps_frames = 0

    print("[INFO] Webcam started. Press q or Esc in the video window to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[WARN] Could not read frame from camera.")
                break

            if not args.no_mirror and isinstance(parse_camera_source(args.camera), int):
                frame = cv2.flip(frame, 1)

            display_frame = resize_for_recognition(frame, args.frame_width)

            if frame_count % process_every == 0:
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                rects = detect_faces(models, rgb_frame, args.upsample)
                face_results = []

                for rect in rects:
                    encoding = compute_face_encoding(models, rgb_frame, rect, args.jitter)
                    name, distance, matched = match_face(
                        known_encodings,
                        known_names,
                        encoding,
                        args.threshold,
                        args.unknown_name,
                    )
                    face_results.append((rect, name, distance, matched))

            for rect, name, distance, matched in face_results:
                label = f"{name} ({distance:.3f})" if distance != float("inf") else name
                draw_face_label(display_frame, rect, label, matched)

            fps_frames += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.time()

            draw_status(display_frame, fps, models.detector_kind)
            cv2.imshow(args.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            frame_count += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("[INFO] Webcam stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
