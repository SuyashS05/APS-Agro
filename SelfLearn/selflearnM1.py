import cv2
from ultralytics import YOLO
import os
import time

# Paths
model_path = "YOLO_Dataset/trained_model1.1/broccoli_model_resume/weights/best.pt"
image_save_path = "F:/Final_Broccolli_Detection_ML/SelfLearn/Img"
annotation_save_path = "F:/Final_Broccolli_Detection_ML/SelfLearn/annotation"

# Load YOLO model
model = YOLO(model_path)

# Confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.75

# Function to scale and format YOLO annotation
def format_yolo_annotation(box, frame_width, frame_height):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2 / frame_width
    y_center = (y_min + y_max) / 2 / frame_height
    width = (x_max - x_min) / frame_width
    height = (y_max - y_min) / frame_height
    return x_center, y_center, width, height

# Select video file
video_path = input("Enter the path of the video file: ")
video_name = os.path.splitext(os.path.basename(video_path))[0]
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

frame_count = 0
os.makedirs(image_save_path, exist_ok=True)
os.makedirs(annotation_save_path, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    # Detect objects in frame
    results = model(frame)
    detected = results[0]
    frame_annotations = []

    for result in detected.boxes.data:
        x_min, y_min, x_max, y_max, confidence, cls = result.tolist()
        cls = int(cls)
        box = (x_min, y_min, x_max, y_max)

        if confidence >= CONFIDENCE_THRESHOLD:
            # Create YOLO annotation format and append
            x_center, y_center, width, height = format_yolo_annotation(box, frame_width, frame_height)
            frame_annotations.append(f"{cls} {x_center} {y_center} {width} {height}")

    # Save frame and annotations if detections are present
    if frame_annotations:
        frame_count += 1
        image_filename = f"{video_name}_frame_{frame_count}.jpg"
        annotation_filename = f"{video_name}_frame_{frame_count}.txt"

        # Save frame
        image_path = os.path.join(image_save_path, image_filename)
        cv2.imwrite(image_path, frame)

        # Save annotations
        annotation_path = os.path.join(annotation_save_path, annotation_filename)
        with open(annotation_path, 'w') as annotation_file:
            annotation_file.write("\n".join(frame_annotations))

    # Display the video
    cv2.imshow('Processed Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
