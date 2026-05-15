import cv2
from ultralytics import YOLO
import numpy as np
import serial
import time

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Thresholds
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.75
TIMEOUT_FRAMES = 10

# Load YOLO model
model_path = "YOLO_Dataset/trained_model1.1/broccoli_model_resume/weights/best.pt"
model = YOLO(model_path)

# # IP Webcam live stream URL
# video_url = "http://192.168.96.135:8080/video"  # Replace with your correct URL
# cap = cv2.VideoCapture(video_url)

# ESP32 Webcam live stream URL
video_url = "http://192.168.45.249:81/stream"
cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"Error: Could not open video stream: {video_url}")
    exit()

frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_with_grid_map.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

UNIT_WIDTH, UNIT_HEIGHT = 800, 800  # Scaled field graph
field_graph = 255 * np.ones((UNIT_HEIGHT, UNIT_HEIGHT, 3), dtype=np.uint8)

def draw_grid(image, rows=10, cols=10, color=(200, 200, 200)):
    """Draw a grid on the image."""
    h, w, _ = image.shape
    row_step, col_step = h // rows, w // cols
    for i in range(0, h, row_step):
        cv2.line(image, (0, i), (w, i), color, 1)
    for j in range(0, w, col_step):
        cv2.line(image, (j, 0), (j, h), color, 1)

def scale_coordinates(x, y, frame_width, frame_height, unit_width, unit_height):
    """Map coordinates from the frame to the field graph."""
    return int(x * unit_width / frame_width), int(y * unit_height / frame_height)

# ESP32/ESP8266 Serial Communication Setup
esp_port = "COM5"  # Replace with the correct port for your ESP
baud_rate = 115200
esp_serial = serial.Serial(esp_port, baud_rate, timeout=1)

def send_command_to_esp(command):
    """Send a command to the ESP32/ESP8266."""
    esp_serial.write((command + '\n').encode())
    time.sleep(0.1)  # Allow time for the ESP to process

object_tracker = {'plant': {}, 'fruit': {}}
timeout_tracker = {'plant': {}, 'fruit': {}}
next_id = {'plant': 1, 'fruit': 1}

total_broccoli_plant_count = 0
total_broccoli_fruit_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    current_frame_objects = {'plant': set(), 'fruit': set()}

    for result in results[0].boxes.data:
        x_min, y_min, x_max, y_max, confidence, cls = result.tolist()
        cls = int(cls)
        box = (x_min, y_min, x_max, y_max)

        if confidence >= CONFIDENCE_THRESHOLD:
            category = 'plant' if cls == 0 else 'fruit' if cls == 1 else None
            if category:
                found = False
                for obj_id, tracked_box in object_tracker[category].items():
                    if calculate_iou(box, tracked_box) > IOU_THRESHOLD:
                        found = True
                        object_tracker[category][obj_id] = box
                        timeout_tracker[category][obj_id] = TIMEOUT_FRAMES
                        break
                if not found:
                    object_tracker[category][next_id[category]] = box
                    timeout_tracker[category][next_id[category]] = TIMEOUT_FRAMES

                    center_x, center_y = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
                    scaled_x, scaled_y = scale_coordinates(center_x, center_y, frame_width, frame_height, UNIT_WIDTH, UNIT_HEIGHT)
                    color = (0, 255, 0) if category == 'plant' else (0, 0, 255)
                    cv2.circle(field_graph, (scaled_x, UNIT_HEIGHT - scaled_y), 5, color, -1)
                    draw_grid(field_graph)  # Draw grid lines

                    if category == 'plant':
                        total_broccoli_plant_count += 1
                        send_command_to_esp("PLANT")  # Blink LED for plant
                    elif category == 'fruit':
                        total_broccoli_fruit_count += 1
                        send_command_to_esp("FRUIT")  # Blink LED for fruit
                    next_id[category] += 1
                current_frame_objects[category].add(box)

    for category in ['plant', 'fruit']:
        for obj_id in list(object_tracker[category].keys()):
            timeout_tracker[category][obj_id] -= 1
            if timeout_tracker[category][obj_id] <= 0:
                del object_tracker[category][obj_id]
                del timeout_tracker[category][obj_id]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated_frame, f'Total Broccoli Plants: {total_broccoli_plant_count}', (10, 40), font, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'Total Broccoli Fruits: {total_broccoli_fruit_count}', (10, 80), font, 1, (0, 255, 0), 2)

    cv2.imshow('Plant Field Mapping', field_graph)
    cv2.imshow('Detection Display', annotated_frame)

    out.write(annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
