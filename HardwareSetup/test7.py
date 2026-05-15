import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, filedialog

# Create a hidden Tkinter root window for file dialog
root = Tk()
root.withdraw()  # Hide the root window

# Open file dialog to select a video
video_path = filedialog.askopenfilename(
    title="Select a Video File",
    filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")]
)

if not video_path:
    print("No video selected. Exiting...")
    exit()

print(f"Selected video: {video_path}")

# Path to your trained model
model_path = "YOLO_Dataset/trained_model1.1/broccoli_model_resume/weights/best.pt"  # Replace with your model file path

# Load the trained YOLO model
model = YOLO(model_path)

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video: {video_path}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Properties - FPS: {fps}, Width: {frame_width}, Height: {frame_height}")

# Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Perform inference
    results = model(frame)

    # Get detection boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [[x1, y1, x2, y2], ...]
    centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        centers.append((center_x, center_y))

    # Create an overlay for annotations
    overlay = frame.copy()

    if len(centers) >= 2:
        centers = np.array(centers)
        # Fit regression for main broccoli row
        poly = np.polyfit(centers[:, 1], centers[:, 0], 1)  # 1st degree (straight line)
        poly_func = np.poly1d(poly)

        # Generate lane coordinates for the main row
        y_coords = np.arange(0, frame_height, 1)  # All y-coordinates
        x_coords_main = poly_func(y_coords).astype(int)

        # Draw the main row (Green)
        for y, x in zip(y_coords, x_coords_main):
            if 0 <= x < frame_width:
                cv2.line(overlay, (x, y), (x, y), (0, 255, 0), 10)  # Green broad line

        # Offset lanes for adjacent rows
        lane_offset = 50  # Adjust for row spacing
        x_coords_left = (x_coords_main - lane_offset).clip(0, frame_width - 1)
        x_coords_right = (x_coords_main + lane_offset).clip(0, frame_width - 1)

        # Draw adjacent rows (Blue)
        for y, x_left, x_right in zip(y_coords, x_coords_left, x_coords_right):
            cv2.line(overlay, (x_left, y), (x_left, y), (255, 0, 0), 10)  # Blue line
            cv2.line(overlay, (x_right, y), (x_right, y), (255, 0, 0), 10)  # Blue line

        # Fill ground soil area between rows (Black)
        for y, x_left, x_right in zip(y_coords, x_coords_left, x_coords_right):
            cv2.line(overlay, (x_left, y), (x_right, y), (0, 0, 0), 2)  # Black fill

    # Blend annotations with the original frame
    alpha = 0.5  # Transparency factor
    annotated_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display the live video
    cv2.imshow("Live Lane Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Processing complete.")
