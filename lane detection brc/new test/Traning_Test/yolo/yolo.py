from ultralytics import YOLO
import torch

# Check if CUDA (GPU) is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model (Change to yolov8s.pt or yolov8m.pt for better accuracy)
model = YOLO("yolov8n.pt")  

# Train YOLO Model
model.train(
    data="F:/lane detection brc/Traning_Test/yolo/config.yaml",
    epochs=100,       # Increased epochs for better training
    imgsz=640,        # Input image size
    batch=16,         # Adjust batch size based on your hardware
    workers=4,        # Set workers for dataloader
    device=device,    # Use GPU if available
    name="yolo_lane_detection",
    project="F:/lane detection brc/runs/detect",
    exist_ok=True,    # Prevent overwriting
    optimizer="SGD",  # Using SGD optimizer (try 'Adam' for better performance)
    lr0=0.01,         # Initial learning rate
    lrf=0.1,          # Final learning rate
    patience=20,      # Stop training early if no improvement
    save_period=10,   # Save model every 10 epochs
    cache=True,       # Cache images for faster training
    augment=True      # Data augmentation for better robustness
)

print("✅ Training Complete! Model saved at:")
print("F:/lane detection brc/runs/detect/yolo_lane_detection/")
