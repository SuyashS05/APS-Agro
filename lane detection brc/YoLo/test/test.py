import cv2
import os
from ultralytics import YOLO

# Load the trained YOLOv8 model
model_path = r"F:\lane detection brc\Traning_Test\yolo\output\yolo_lane_detection\weights\best.pt"
model = YOLO(model_path)

# Path to the folder containing videos
video_folder = r"F:\lane detection brc\YoLo\test\vedio"

# Get list of video files
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

if not video_files:
    print("No video files found in the specified folder.")
    exit()

# Process each video in the folder
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        continue

    print(f"Processing video: {video_file}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Perform YOLO lane detection
        results = model(frame)

        # Draw detections on the frame
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = box.conf[0]

                # Define label names (modify as per your dataset)
                class_names = ["Main Row", "Adjacent Row", "APS Path"]
                label = f"{class_names[class_id]}: {confidence:.2f}"

                # Define colors
                colors = [(255, 255, 255), (255, 0, 0), (0, 0, 255)]  # White, Blue, Red

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_id], 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_id], 2)

        # Display the frame with detections
        cv2.imshow("YOLO Lane Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()
