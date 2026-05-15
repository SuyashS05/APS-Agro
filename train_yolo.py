import os
import sys
import torch
import yaml

# Add YOLOv5 package to Python path
yolov5_path = "C:/Users/ASHOK/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages/yolov5"
sys.path.append(yolov5_path)

from yolov5 import train

def main():
    # Load data.yaml file
    with open('data.yaml', 'r') as file:
        data_config = yaml.safe_load(file)

    # Ensure paths to images and labels exist
    assert os.path.exists(data_config['train']), f"Training images path {data_config['train']} does not exist"
    assert os.path.exists(data_config['val']), f"Validation images path {data_config['val']} does not exist"

    # Check that labels paths exist
    train_labels_path = data_config['train'].replace('images', 'labels')
    val_labels_path = data_config['val'].replace('images', 'labels')
    assert os.path.exists(train_labels_path), f"Training labels path {train_labels_path} does not exist"
    assert os.path.exists(val_labels_path), f"Validation labels path {val_labels_path} does not exist"

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set training parameters
    data = 'data.yaml'  # path to data.yaml
    weights = 'yolov5s.pt'  # initial weights
    epochs = 50  # number of training epochs
    batch_size = 16  # batch size

    # Train the model
    train.run(data=data, weights=weights, epochs=epochs, batch_size=batch_size, device=device)

if __name__ == '__main__':
    main()
