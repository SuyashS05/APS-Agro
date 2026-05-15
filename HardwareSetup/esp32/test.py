import cv2
from ultralytics import YOLO

# Load YOLO model
model_path = "YOLO_Dataset/trained_model1.1/broccoli_model_resume/weights/best.pt"
model = YOLO(model_path)

# ESP32-CAM live stream URL
video_url = "http://192.168.45.249:81/stream"
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default to 30 FPS if not detected

# Define the codec and create VideoWriter object
output_filename = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame")
        break

    # Run YOLO detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Save the processed frame to the video file
    out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow("ESP32-CAM Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_filename}")



# import cv2
# from ultralytics import YOLO

# # Load YOLO model
# model_path = "YOLO_Dataset/trained_model1.1/broccoli_model_resume/weights/best.pt"
# model = YOLO(model_path)

# # ESP32-CAM live stream URL
# video_url = "http://192.168.45.249:81/stream"
# cap = cv2.VideoCapture(video_url)

# if not cap.isOpened():
#     print("Error: Could not open video stream")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Couldn't read frame")
#         break

#     # Run YOLO detection
#     results = model(frame)
#     annotated_frame = results[0].plot()

#     # Display the annotated frame
#     cv2.imshow("ESP32-CAM Object Detection", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
