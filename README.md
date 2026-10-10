# APS Agro: Broccoli and Field Lane Detection

This repository contains computer-vision experiments for agricultural field monitoring. The main thread of work is detecting broccoli plants or heads in images and video, with related experiments in crop-lane detection, dataset preparation, and ESP32 camera integration.

The repository is an experiment archive rather than a packaged application: scripts have separate dependencies and several still use machine-specific paths. Start with the YOLO inference script below and adapt paths and hardware settings to your setup.

## What is included

- **Broccoli detection:** YOLO training and inference scripts, a trained weights file (`best.pt`), video examples, and datasets/annotations under `brocolli_detection/`, `Dataset/`, and `SelfLearn/`.
- **Lane detection:** YOLO, LaneNet/U-Net, and DeepLab experiments and data under `lane detection brc/` and `final_Broccoli_Lane_detection_ML/`.
- **Dataset utilities:** scripts for drawing YOLO annotations, augmentation, cropping, resizing, renaming, and train/validation partitioning.
- **Hardware experiments:** ESP32-CAM stream inference and serial signaling prototypes in `HardwareSetup/`.
- **Notebooks and artifacts:** preprocessing/training notebooks, sample outputs, spreadsheets, model weights, and recorded videos.

## Visual results

The fine-tuned model run under `Final_Broccolli_Detection_ML/YOLO_Dataset/trained_model2.0/fine_tuned_model_broccoli2.0/` includes Ultralytics training previews. The label summary shows three classes: **Broccoli Plant**, **Broccoli Fruit**, and **Soil/Land**. Its plotted counts are imbalanced, with plant instances most common and fruit instances least common.

| Label distribution and box locations | Training batch preview |
| --- | --- |
| ![Label counts, box locations, and normalized box dimensions](docs/assets/labels.jpg) | ![Training examples with labeled boxes](docs/assets/train_batch0.jpg) |

An additional augmented training batch is available as [train_batch2.jpg](docs/assets/train_batch2.jpg). These are training and label-inspection artifacts, not evaluation metrics.

### Annotated field video

[![Preview of the annotated field video; click to open the MP4](docs/assets/output_annotated3_preview.jpg)](output_annotated3.mp4)

Click the preview to open `output_annotated3.mp4`. The clip is approximately 10 seconds long (303 frames at 30 fps, 478 × 850) and overlays detected broccoli plants and soil/land with confidence scores. It is an example output, not a benchmark or accuracy claim.

## Broccoli inference

The root `test.py` loads `best.pt` from the repository root. It offers three modes: select an image, use the default webcam, or select a video. Image results are displayed and saved by Ultralytics; video and webcam detections are shown in a window. Press **q** to stop a live video or webcam run.

Create a Python environment and install the packages used by this script:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python matplotlib
```

Run from the repository root so `best.pt` resolves correctly:

```bash
python test.py
```

This workflow requires a graphical desktop for the file picker and OpenCV windows. For headless environments, adapt the script to accept an input path and save predictions without calling `show()`/`imshow()`.

## Training and datasets

`train_yolo.py` is an older YOLOv5 training entry point. It expects a YOLOv5 Python package, PyTorch, and PyYAML, and chooses CUDA when available. The current root `data.yaml` contains absolute paths from a previous Windows machine (`E:/brocolli_detection/...`), so edit its `train` and `val` values before training. The referenced image directories should have corresponding `labels/train` and `labels/val` folders in the layout expected by the training script.

The root configuration declares `nc: 1` alongside three class names. Confirm the intended classes and make `nc` match the number of entries in `names` before using it. The three-class fine-tuned experiment shown in the visual artifacts is a separate run and does not use this root configuration as currently written. The separate YOLO lane-detection configuration and scripts are under `lane detection brc/YoLo/`.

Dataset preparation scripts at the root include `annotation.py`, `Augmentation.py`, `imageCrop.py`, `partition.py`, and several resize/rename utilities. These are standalone utilities with paths and assumptions embedded in their source; inspect and update those paths for your dataset before running them. `partition.py` expects images in `dataset/resized_plant` and labels in `dataset/annotations` and writes a train/validation structure under `dataset/`.

## ESP32-CAM experiments

`HardwareSetup/esp32/test.py` reads an ESP32-CAM HTTP stream and writes an annotated video. `plantCount.py` and `livecamPlantCount.py` add object counting, a grid visualization, and serial commands to an ESP board. Before running these scripts, update the model/video paths, camera stream URL, and serial port in the source. They require OpenCV, Ultralytics, NumPy, and (for serial-enabled scripts) pyserial. The current scripts are prototypes and assume a configured camera and compatible firmware listening for the emitted commands.

## Repository map

```text
.
├── best.pt                         # Root YOLO model used by test.py
├── test.py                         # Image, webcam, and video inference
├── train_yolo.py                   # YOLOv5 training experiment
├── data.yaml                       # Root broccoli dataset configuration
├── annotation.py / partition.py    # Dataset annotation and splitting tools
├── brocolli_detection/             # Broccoli data, annotations, and videos
├── Dataset/                        # Additional image dataset
├── SelfLearn/                      # Self-learning images and labels
├── lane detection brc/             # Lane detection experiments and data
├── final_Broccoli_Lane_detection_ML/ # Later experiments, GPS sheets, data
├── HardwareSetup/                  # ESP32/ESP32-CAM prototypes
└── Colab Notebooks/                # Notebook-based preprocessing/training
```

## Notes

- There is no single project-wide dependency file; install dependencies for the specific script you intend to run.
- Many datasets and output videos are large. Check available storage before copying or processing them.
- Paths, class mappings, model versions, and dataset layouts vary among experiments. Verify them against the selected script before training or inference.
- `best.pt` is the provided model artifact; its metrics and training provenance are not documented in the repository.
