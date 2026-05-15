from ultralytics import YOLO
import torch

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Try "yolov8s.pt" or "yolov8m.pt" for better accuracy

# Train the model
model.train(
    data="/content/drive/MyDrive/lane_detection_brc/YoLo/data.yaml",
    epochs=100,       # Adjust as needed
    imgsz=640,        # Image size
    batch=16,         # Adjust based on GPU capability
    workers=2,        # Number of CPU threads
    device=device,    # Auto GPU/CPU selection
    name="yolo_lane_detection",
    project="/content/drive/MyDrive/lane_detection_brc/runs/detect",
    exist_ok=True,    # Prevent overwriting
    optimizer="SGD",  # Use SGD optimizer (Adam also possible)
    lr0=0.01,         # Initial learning rate
    lrf=0.1,          # Final learning rate
    patience=20,      # Stops training early if no improvement
    save_period=10,   # Saves model every 10 epochs
    cache=True,       # Caches images for faster training
    augment=True      # Data augmentation enabled
)

print("✅ Training Complete! Model saved at:")
print("/content/drive/MyDrive/lane_detection_brc/runs/detect/yolo_lane_detection/")
