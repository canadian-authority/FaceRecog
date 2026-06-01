from __future__ import annotations

import argparse
import csv
import pickle
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

try:
    import dlib
except ImportError:
    dlib = None

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENCODINGS = BASE_DIR / "encodings.pickle"
DEFAULT_SHAPE_PREDICTOR = BASE_DIR / "shape_predictor_68_face_landmarks.dat"
DEFAULT_FACE_MODEL = BASE_DIR / "dlib_face_recognition_resnet_model_v1.dat"
DEFAULT_ATTENDANCE = BASE_DIR / "attendance.csv"
DEFAULT_SNAPSHOT_DIR = BASE_DIR / "attendance_screenshots"
NON_ATTENDANCE_NAMES = {"", "unknown", "face"}


class FrameSource(Protocol):
    def read(self):
        ...

    def release(self) -> None:
        ...


class PiCameraSource:
    def __init__(self, width: int, height: int, fps: int) -> None:
        if Picamera2 is None:
            raise RuntimeError(
                "picamera2 is not installed. Install it on Raspberry Pi OS with: "
                "sudo apt install python3-picamera2"
            )

        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"},
            controls={"FrameRate": fps},
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(1.0)

    def read(self):
        return True, self.camera.capture_array()

    def release(self) -> None:
        self.camera.stop()
        self.camera.close()


class OpenCVCameraSource:
    def __init__(self, source: int | str, width: int, height: int, fps: int) -> None:
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera/source: {source}")

    def read(self):
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


@dataclass(frozen=True)
class Recognition:
    left: int
    top: int
    right: int
    bottom: int
    name: str
    distance: float | None = None
    hits: int = 0


def is_attendance_name(name: str) -> bool:
    return name.strip().casefold() not in NON_ATTENDANCE_NAMES


def format_match_score(distance: float | None) -> str:
    if distance is None or distance == float("inf"):
        return ""
    return f"{distance:.3f}"


class AttendanceBook:
    HEADER = ["Name", "Probability", "Date&Time"]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.present: set[str] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()
        self._load_existing()

    def _ensure_file(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._write_rows([])
            return

        with self.path.open("r", newline="", encoding="utf-8") as file:
            lines = file.readlines()

        if not lines:
            self._write_rows([])
            return

        header = self._parse_csv_line(lines[0].strip())
        normalized_header = [value.strip().casefold() for value in header]
        expected_header = [value.casefold() for value in self.HEADER]
        if normalized_header == expected_header:
            return

        has_header = bool(normalized_header) and normalized_header[0] == "name"
        rows = [
            self._normalize_existing_row(line.strip())
            for line in (lines[1:] if has_header else lines)
        ]
        self._write_rows([row for row in rows if row is not None])

    def _write_rows(self, rows: list[list[str]]) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(self.HEADER)
            writer.writerows(rows)

    def _parse_csv_line(self, line: str) -> list[str]:
        delimiter = ";" if ";" in line else ","
        return next(csv.reader([line], delimiter=delimiter), [])

    def _looks_like_date(self, value: str) -> bool:
        value = value.strip()
        return len(value) >= 10 and value[4] == "-" and value[7] == "-"

    def _looks_like_score(self, value: str) -> bool:
        try:
            float(value)
        except ValueError:
            return False
        return ":" not in value

    def _normalize_existing_row(self, line: str) -> list[str] | None:
        row = self._parse_csv_line(line)
        if not row:
            return None

        name = row[0].strip()
        if name.casefold() == "name" or not is_attendance_name(name):
            return None

        values = [value.strip() for value in row[1:]]
        probability = ""
        date_time = ""

        if len(values) >= 2 and self._looks_like_date(values[1]):
            probability = values[0]
            if self._looks_like_score(values[0]):
                date_time = values[1]
            else:
                probability = ""
                date_time = f"{values[1]} {values[0]}".strip()
        elif len(values) >= 2 and self._looks_like_date(values[0]):
            date_time = f"{values[0]} {values[1]}".strip()
        elif values:
            date_time = values[0]

        return [name, probability, date_time]

    def _extract_date(self, values: list[str]) -> str:
        for value in values:
            if self._looks_like_date(value):
                return value[:10]
        return self.today

    def _parse_attendance_line(self, line: str) -> tuple[str, str] | None:
        row = self._parse_csv_line(line)
        if not row:
            return None

        name = row[0].strip()
        if name.casefold() == "name" or not is_attendance_name(name):
            return None

        return name, self._extract_date(row[1:])

    def _load_existing(self) -> None:
        with self.path.open("r", newline="", encoding="utf-8") as file:
            for line in file:
                parsed = self._parse_attendance_line(line.strip())
                if parsed is None:
                    continue

                name, date = parsed
                if date == self.today:
                    self.present.add(name)

    def mark(self, name: str, distance: float | None = None) -> bool:
        name = name.strip()
        if not is_attendance_name(name) or name in self.present:
            return False

        now = datetime.now()
        self._ensure_file()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        probability = format_match_score(distance)
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow([name, probability, date_time])

        self.present.add(name)
        score_label = f" ({probability})" if probability else ""
        print(f"[ATTENDANCE] Marked {name}{score_label} at {date_time}")
        return True


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return BASE_DIR / expanded


def default_attendance_file() -> Path:
    return DEFAULT_ATTENDANCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast Raspberry Pi 5 face recognition attendance using dlib HOG.",
    )
    parser.add_argument(
        "--camera",
        choices=("auto", "picamera2", "opencv"),
        default="auto",
        help="Camera API to use. Default: auto.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="OpenCV camera index, video file, or stream URL when --camera opencv is used.",
    )
    parser.add_argument("--width", type=int, default=640, help="Capture width. Default: 640.")
    parser.add_argument("--height", type=int, default=480, help="Capture height. Default: 480.")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS. Default: 30.")
    parser.add_argument(
        "--process-width",
        type=int,
        default=480,
        help="Width used for detection/recognition. Lower is faster. Default: 480.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=3,
        help="Run recognition every N frames. Higher is faster but less responsive. Default: 3.",
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=0,
        help="dlib HOG upsample count. Use 1 for small/far faces, slower. Default: 0.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.52,
        help="Max face distance for a match. Lower is stricter. Default: 0.52.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.03,
        help="Required gap between best and second-best person. Use 0 to disable. Default: 0.03.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=12,
        help="Processed frames kept for verification. Default: 12.",
    )
    parser.add_argument(
        "--required-hits",
        type=int,
        default=1,
        help="Recognition hits required before attendance is marked. Use 1 to mark as soon as a face is recognized. Default: 1.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=640,
        help="Resize preview to this width. Use 0 to keep camera size. Default: 640.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without a preview window.")
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only draw dlib face boxes; do not load encodings or mark attendance.",
    )
    parser.add_argument(
        "--encodings",
        type=Path,
        default=DEFAULT_ENCODINGS,
        help="Path to encodings.pickle.",
    )
    parser.add_argument(
        "--shape-predictor",
        type=Path,
        default=DEFAULT_SHAPE_PREDICTOR,
        help="Path to dlib shape predictor model.",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=DEFAULT_FACE_MODEL,
        help="Path to dlib face recognition model.",
    )
    parser.add_argument(
        "--attendance-file",
        type=Path,
        default=None,
        help="CSV attendance file. Default: attendance.csv beside this script.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="Folder for snapshots when attendance is marked.",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Do not save snapshots when attendance is marked.",
    )
    parser.add_argument(
        "--cv-threads",
        type=int,
        default=2,
        help="OpenCV worker threads. Use 0 to keep OpenCV default. Default: 2.",
    )
    parser.add_argument(
        "--window-name",
        default="Pi 5 Face Recognition Attendance",
        help="OpenCV preview window title.",
    )
    return parser.parse_args()


def create_frame_source(args: argparse.Namespace) -> FrameSource:
    if args.camera == "picamera2":
        return PiCameraSource(args.width, args.height, args.fps)

    if args.camera == "opencv":
        return OpenCVCameraSource(parse_source(args.source), args.width, args.height, args.fps)

    if Picamera2 is not None:
        try:
            return PiCameraSource(args.width, args.height, args.fps)
        except Exception as exc:
            print(f"[WARN] Picamera2 failed, trying OpenCV instead: {exc}")

    return OpenCVCameraSource(parse_source(args.source), args.width, args.height, args.fps)


def validate_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def load_resources(args: argparse.Namespace):
    if dlib is None:
        raise SystemExit("dlib is not installed in this Python environment.")

    detector = dlib.get_frontal_face_detector()
    if args.detect_only:
        return detector, None, None, np.empty((0, 128), dtype=np.float32), []

    encodings_path = resolve_path(args.encodings)
    predictor_path = resolve_path(args.shape_predictor)
    face_model_path = resolve_path(args.face_model)
    validate_file(encodings_path, "Encodings file")
    validate_file(predictor_path, "Shape predictor model")
    validate_file(face_model_path, "Face recognition model")

    with encodings_path.open("rb") as file:
        data = pickle.load(file)

    known_encodings = np.asarray(data["encodings"], dtype=np.float32)
    known_names = [str(name) for name in data["names"]]
    if known_encodings.ndim != 2 or known_encodings.shape[1] != 128:
        raise SystemExit(f"Expected 128D encodings, got shape {known_encodings.shape}.")

    predictor = dlib.shape_predictor(str(predictor_path))
    facerec = dlib.face_recognition_model_v1(str(face_model_path))
    print(f"[INFO] Loaded {len(known_names)} encodings for {len(set(known_names))} people.")
    return detector, predictor, facerec, known_encodings, known_names


def resize_for_processing(frame, process_width: int):
    height, width = frame.shape[:2]
    if process_width <= 0 or width <= process_width:
        return frame, 1.0, 1.0

    scale = process_width / float(width)
    process_height = max(1, int(height * scale))
    resized = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
    return resized, width / float(process_width), height / float(process_height)


def resize_for_display(frame, display_width: int):
    height, width = frame.shape[:2]
    if display_width <= 0 or width <= display_width:
        return frame, 1.0, 1.0

    scale = display_width / float(width)
    display_height = max(1, int(height * scale))
    resized = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    return resized, display_width / float(width), display_height / float(height)


def clip_rect(rect, width: int, height: int) -> tuple[int, int, int, int] | None:
    left = max(0, rect.left())
    top = max(0, rect.top())
    right = min(width - 1, rect.right())
    bottom = min(height - 1, rect.bottom())
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def match_face(
    known_encodings: np.ndarray,
    known_names: list[str],
    face_encoding: np.ndarray,
    threshold: float,
    margin: float,
) -> tuple[str, float]:
    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
    best_index = int(np.argmin(distances))
    best_name = known_names[best_index]
    best_distance = float(distances[best_index])

    best_by_name: dict[str, float] = {}
    for name, distance in zip(known_names, distances):
        distance = float(distance)
        if name not in best_by_name or distance < best_by_name[name]:
            best_by_name[name] = distance

    other_distances = [
        distance for name, distance in best_by_name.items() if name != best_name
    ]
    second_best = min(other_distances, default=float("inf"))

    if best_distance <= threshold and second_best - best_distance >= margin:
        return best_name, best_distance
    return "Unknown", best_distance


def recognize_frame(
    frame,
    detector,
    predictor,
    facerec,
    known_encodings: np.ndarray,
    known_names: list[str],
    args: argparse.Namespace,
    history: deque[set[str]],
) -> list[Recognition]:
    process_frame, scale_x, scale_y = resize_for_processing(frame, args.process_width)
    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width = rgb_frame.shape[:2]

    recognitions: list[Recognition] = []
    names_in_frame: set[str] = set()
    rects = detector(rgb_frame, args.upsample)

    for rect in rects:
        clipped = clip_rect(rect, frame_width, frame_height)
        if clipped is None:
            continue

        left, top, right, bottom = clipped
        name = "Face"
        distance = None

        if not args.detect_only:
            shape = predictor(rgb_frame, dlib.rectangle(left, top, right, bottom))
            face_encoding = np.asarray(
                facerec.compute_face_descriptor(rgb_frame, shape),
                dtype=np.float32,
            )
            name, distance = match_face(
                known_encodings,
                known_names,
                face_encoding,
                args.threshold,
                args.margin,
            )
            if name != "Unknown":
                names_in_frame.add(name)

        recognitions.append(
            Recognition(
                left=int(left * scale_x),
                top=int(top * scale_y),
                right=int(right * scale_x),
                bottom=int(bottom * scale_y),
                name=name,
                distance=distance,
            )
        )

    if not args.detect_only:
        history.append(names_in_frame)
        hit_counts = {
            name: sum(1 for frame_names in history if name in frame_names)
            for name in names_in_frame
        }
        recognitions = [
            Recognition(
                item.left,
                item.top,
                item.right,
                item.bottom,
                item.name,
                item.distance,
                hit_counts.get(item.name, 0),
            )
            for item in recognitions
        ]

    return recognitions


def draw_recognitions(frame, recognitions: list[Recognition], present: set[str], args: argparse.Namespace) -> None:
    for item in recognitions:
        score = format_match_score(item.distance)
        score_label = f" ({score})" if score else ""

        if item.name == "Unknown":
            color = (0, 0, 255)
            label = f"Unknown{score_label}"
        elif item.name == "Face":
            color = (0, 200, 0)
            label = "Face"
        elif item.name in present:
            color = (0, 200, 0)
            label = f"{item.name}{score_label} Present"
        else:
            color = (0, 215, 255)
            if args.required_hits <= 1:
                label = f"{item.name}{score_label}"
            else:
                label = f"{item.name}{score_label} {item.hits}/{args.required_hits}"

        cv2.rectangle(frame, (item.left, item.top), (item.right, item.bottom), color, 2)
        label_y = max(22, item.top - 8)
        cv2.putText(
            frame,
            label,
            (item.left, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def draw_status(frame, fps: float, recognitions: list[Recognition], attendance: AttendanceBook | None) -> None:
    if attendance is None:
        text = f"Faces: {len(recognitions)} | FPS: {fps:.1f} | q/Esc quit"
    else:
        text = f"Faces: {len(recognitions)} | Present: {len(attendance.present)} | FPS: {fps:.1f} | q/Esc quit"

    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )


def save_snapshot(snapshot_dir: Path, frame, name: str) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = snapshot_dir / f"{safe_name}_{stamp}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def mark_verified_people(
    recognitions: list[Recognition],
    attendance: AttendanceBook,
    frame,
    snapshot_dir: Path,
    save_snapshots: bool,
    required_hits: int,
) -> None:
    marked_names: set[str] = set()
    for item in recognitions:
        if (
            not is_attendance_name(item.name)
            or item.name in marked_names
            or item.hits < required_hits
        ):
            continue
        marked_names.add(item.name)
        if attendance.mark(item.name, item.distance) and save_snapshots:
            snapshot_path = save_snapshot(snapshot_dir, frame, item.name)
            print(f"[SNAPSHOT] Saved {snapshot_path}")


def main() -> int:
    args = parse_args()
    if args.process_every < 1:
        raise SystemExit("--process-every must be at least 1.")
    if args.required_hits < 1:
        raise SystemExit("--required-hits must be at least 1.")
    if args.history_size < args.required_hits:
        raise SystemExit("--history-size must be greater than or equal to --required-hits.")

    if args.cv_threads > 0:
        cv2.setNumThreads(args.cv_threads)
    cv2.setUseOptimized(True)

    detector, predictor, facerec, known_encodings, known_names = load_resources(args)
    attendance = None if args.detect_only else AttendanceBook(
        resolve_path(args.attendance_file) if args.attendance_file else default_attendance_file()
    )
    snapshot_dir = resolve_path(args.snapshot_dir)
    if attendance is not None:
        print(f"[INFO] Attendance file: {attendance.path}")
        print(f"[INFO] Already marked today: {len(attendance.present)}")

    source = create_frame_source(args)
    history: deque[set[str]] = deque(maxlen=args.history_size)
    recognitions: list[Recognition] = []
    frame_count = 0
    fps = 0.0
    fps_start = time.time()
    fps_frames = 0
    last_headless_log = 0.0

    print("[INFO] Started. Press q/Esc in the preview window, or Ctrl+C in the terminal, to quit.")
    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("[WARN] Could not read frame from camera.")
                time.sleep(0.02)
                continue

            if frame_count % args.process_every == 0:
                recognitions = recognize_frame(
                    frame,
                    detector,
                    predictor,
                    facerec,
                    known_encodings,
                    known_names,
                    args,
                    history,
                )
                if attendance is not None:
                    mark_verified_people(
                        recognitions,
                        attendance,
                        frame,
                        snapshot_dir,
                        not args.no_snapshots,
                        args.required_hits,
                    )

            frame_count += 1
            fps_frames += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.time()

            if args.headless:
                now = time.time()
                if now - last_headless_log >= 1.0:
                    names = ", ".join(item.name for item in recognitions) or "none"
                    present_count = len(attendance.present) if attendance is not None else 0
                    print(f"[INFO] FPS: {fps:.1f} | Seen: {names} | Present: {present_count}")
                    last_headless_log = now
                continue

            display, display_scale_x, display_scale_y = resize_for_display(frame.copy(), args.display_width)
            scaled_recognitions = [
                Recognition(
                    int(item.left * display_scale_x),
                    int(item.top * display_scale_y),
                    int(item.right * display_scale_x),
                    int(item.bottom * display_scale_y),
                    item.name,
                    item.distance,
                    item.hits,
                )
                for item in recognitions
            ]
            draw_recognitions(display, scaled_recognitions, attendance.present if attendance else set(), args)
            draw_status(display, fps, scaled_recognitions, attendance)
            cv2.imshow(args.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                snapshot_path = save_snapshot(snapshot_dir, frame, "manual")
                print(f"[SNAPSHOT] Saved {snapshot_path}")
    except KeyboardInterrupt:
        print("\n[INFO] Stopping after Ctrl+C.")
    finally:
        source.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print("\n[INFO] Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
