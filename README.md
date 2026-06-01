# FaceRecog

FaceRecog is a dlib/OpenCV face recognition project. It takes labeled face
images, converts each known face into a numeric face embedding, saves those
embeddings in `encodings.pickle`, and then uses that file to recognize matching
faces in folders of images or in a live webcam stream.

The repository includes the dlib model files needed for detection, landmark
prediction, and face embedding generation. It does not include private training
images or generated face encodings; create those locally from your own dataset.

## What The System Does

The system has two main phases:

1. Build known face encodings.
2. Recognize faces using those saved encodings.

During the build phase, the script scans a labeled dataset folder. Each
subfolder name becomes the person's label. For every usable image, the system
detects a face, finds facial landmarks, converts the face into a 128-dimensional
embedding, and saves the embedding with its label.

During the recognition phase, the system detects faces in a new image or webcam
frame, computes a fresh 128-dimensional embedding for each detected face, and
compares that embedding against the saved known embeddings. If the closest
saved embedding is within the configured distance threshold, the face is labeled
as that person. Otherwise, it is labeled as `Unknown`.

## Project Files

```text
FaceRecog/
  build_encodings.py       Build encodings.pickle from labeled face images
  recognize_folder.py      Recognize faces in every image under a folder
  recognize_webcam.py      Recognize faces from a webcam or video source
  facerec_core.py          Shared model loading, detection, encoding, and matching logic
  requirements.txt         Python dependencies
  *.dat                    dlib model files used by the pipeline
```

The expected dlib model files are:

```text
dlib_face_recognition_resnet_model_v1.dat
mmod_human_face_detector.dat
shape_predictor_68_face_landmarks.dat
```

## Core Models

`mmod_human_face_detector.dat` is the CNN face detector. It finds face bounding
boxes in an image. This detector is usually more accurate than the HOG detector,
especially for harder angles and lighting, but it is slower without GPU support.

`shape_predictor_68_face_landmarks.dat` finds 68 landmark points on each
detected face. These landmarks describe key facial geometry such as the eyes,
nose, mouth, and jawline. The recognizer uses them to align and describe the
face consistently.

`dlib_face_recognition_resnet_model_v1.dat` converts an aligned face into a
128-dimensional vector. Similar faces should produce vectors that are close
together. Different faces should produce vectors that are farther apart.

## Install Dependencies

Install the Python packages:

```powershell
cd FaceRecog
python -m pip install -r requirements.txt
```

If you already have a CUDA-enabled dlib build installed, avoid replacing it
unnecessarily. You can check CUDA support with:

```powershell
python -c "import dlib; print('CUDA:', dlib.DLIB_USE_CUDA)"
```

`CUDA: True` means the CNN detector can use the GPU.

## Prepare A Training Dataset

Create a local dataset folder with one subfolder per person:

```text
train_faces/
  PersonA/
    image_0.jpg
    image_1.jpg
  PersonB/
    image_0.jpg
    image_1.jpg
```

The folder name is the label that will be saved in `encodings.pickle`. For
example, images inside `train_faces/PersonA/` will be saved with the label
`PersonA`.

Use clear images where the face is visible. A few varied images per person are
usually better than many near-identical images. Good variation includes changes
in lighting, head angle, expression, camera distance, and background.

By default, `build_encodings.py` skips images with multiple detected faces. This
prevents the wrong face from being saved under the folder label. If an image
intentionally contains multiple faces and every face should receive the same
label, pass `--allow-multiple`.

## Build Encodings

Run:

```powershell
python build_encodings.py --dataset ".\train_faces" --output "encodings.pickle" --models-dir "."
```

What happens internally:

1. The script loads the dlib detector, landmark predictor, and recognition model.
2. It walks through every supported image file under the dataset folder.
3. It reads each image with OpenCV.
4. It converts the image from BGR to RGB because dlib expects RGB input.
5. It detects face bounding boxes.
6. It skips images with no faces.
7. It skips images with multiple faces unless `--allow-multiple` is used.
8. It predicts 68 landmarks for each accepted face.
9. It computes a 128-dimensional face embedding.
10. It stores each embedding and the folder-derived name in `encodings.pickle`.

The saved pickle contains a dictionary with:

```text
encodings: numeric face vectors
names: matching labels for those vectors
```

## Recognize Faces In A Folder

After `encodings.pickle` exists, run:

```powershell
python recognize_folder.py --input ".\path\to\images" --encodings "encodings.pickle" --models-dir "."
```

The folder recognizer scans all supported images under the input folder. For
each image, it detects every face, computes each face embedding, compares it to
the known embeddings, draws a label on the image, and writes results to disk.

Outputs:

```text
recognized_output/
recognition_results.csv
```

`recognized_output/` contains annotated copies of the input images. Matched
faces are drawn in green. Unknown faces are drawn in red.

`recognition_results.csv` contains one row per detected face, plus rows for
images where no face was found or the image could not be read. Useful columns
include the image path, face index, predicted name, match distance, match
status, bounding box coordinates, and annotated output image path.

## Recognize Faces From A Webcam

After `encodings.pickle` exists, run:

```powershell
python recognize_webcam.py --detector cnn
```

The webcam recognizer opens a camera stream, processes frames, and overlays
labels on the live preview. Press `q` or Esc in the video window to stop.

To reduce work, the webcam script does recognition every N frames and reuses the
last labels between processed frames. The default is every 2 frames:

```powershell
python recognize_webcam.py --process-every 2
```

Use `--frame-width` to resize frames before recognition. Smaller frames are
faster but may reduce accuracy for small or distant faces:

```powershell
python recognize_webcam.py --frame-width 640
```

## Matching Logic

Face matching is based on Euclidean distance between embeddings. For each new
face, the script computes the distance to every saved known encoding and chooses
the closest one.

If the closest distance is less than or equal to `--threshold`, the face is
treated as a match. If it is above the threshold, the face is labeled as
`Unknown`.

Default threshold:

```text
0.5
```

Lower thresholds are stricter and reduce false matches, but they may reject
valid faces. Higher thresholds are more permissive and may recognize more valid
faces, but they can increase false matches.

Examples:

```powershell
python recognize_folder.py --threshold 0.45
python recognize_folder.py --threshold 0.6
```

## Detector Choices

The project supports three detector modes:

```text
auto
cnn
hog
```

`auto` uses the CNN detector when `mmod_human_face_detector.dat` is available.
Otherwise, it falls back to dlib's built-in HOG detector.

`cnn` forces the CNN detector. This is usually better for accuracy, but slower
on CPU.

`hog` forces the HOG detector. This is usually faster on CPU, but may miss more
faces.

Examples:

```powershell
python recognize_folder.py --detector hog
python recognize_folder.py --detector cnn
```

## Useful Options

Increase `--upsample` when faces are small in the image:

```powershell
python recognize_folder.py --upsample 1
```

Increase `--jitter` during encoding to make embeddings more stable. This is
slower, but can improve quality:

```powershell
python build_encodings.py --jitter 20
```

Use `--unknown-name` to change the displayed label for unmatched faces:

```powershell
python recognize_folder.py --unknown-name "Not enrolled"
```

## Recreating The System From Scratch

1. Clone the repository.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Add your own local training dataset in the `train_faces/` format.
4. Build `encodings.pickle` with `build_encodings.py`.
5. Run `recognize_folder.py` or `recognize_webcam.py`.

The dlib `.dat` model files are included in the repository. The private
training dataset and generated `encodings.pickle` are intentionally ignored, so
each user can generate their own encodings from their own images.

## Privacy Note

Face images and face encodings are biometric data. Keep `train_faces/` and
`encodings.pickle` private unless every person in the dataset has agreed to
their data being shared. The `.gitignore` file excludes those local files by
default.
