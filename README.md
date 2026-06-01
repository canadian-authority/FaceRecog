# FaceRecog

Folder-based face recognition using local dlib models and an `encodings.pickle`
file.

## Files Expected

Run commands from the project root:

```text
FaceRecog
```

Expected model files in the project root:

```text
dlib_face_recognition_resnet_model_v1.dat
mmod_human_face_detector.dat
shape_predictor_68_face_landmarks.dat
```

## Install Dependencies

Use the Python environment where your CUDA-enabled `dlib` is already installed.
To avoid replacing that build, install only the missing supporting packages if
needed:

```powershell
cd FaceRecog
python -m pip install numpy opencv-python
```

Verify that Python sees the CUDA build:

```powershell
python -c "import dlib; print('CUDA:', dlib.DLIB_USE_CUDA)"
```

`CUDA: True` means the CNN detector can use the GPU.

## Recognize Faces From Webcam

After `encodings.pickle` exists in the project root, run:

```powershell
python recognize_webcam.py --detector cnn
```

Press `q` or Esc in the webcam window to stop.

## Recognize Faces In A Folder

Run this from:

```powershell
cd FaceRecog
```

To test against a local `facesm6` folder:

```powershell
python recognize_folder.py --input ".\facesm6" --models-dir "."
```

Outputs:

```text
recognized_output\
recognition_results.csv
```

Recognition also needs `encodings.pickle`. If it is not in the project root yet,
run the build command below first. Each row in `recognition_results.csv`
contains the image path, face box, predicted name, match distance, and status.

## Build Or Rebuild Encodings

The training folder should have one folder per person:

```text
facesm6\
  PersonA\
    image_0.jpg
  PersonB\
    image_0.jpg
```

Rebuild the pickle:

```powershell
python build_encodings.py --dataset ".\facesm6" --output "encodings.pickle" --models-dir "."
```

Then recognize using the rebuilt file:

```powershell
python recognize_folder.py --input ".\path\to\images" --encodings "encodings.pickle" --models-dir "."
```

## Useful Options

Use CPU-friendly detection:

```powershell
python recognize_folder.py --detector hog
```

Use stricter matching:

```powershell
python recognize_folder.py --threshold 0.45
```

Improve small-face detection:

```powershell
python recognize_folder.py --upsample 1
```
