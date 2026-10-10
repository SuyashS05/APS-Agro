import os
import shutil
import random
from pathlib import Path

def create_folders(base_path):
    # Create directory structure
    paths = [
        base_path / 'images' / 'train',
        base_path / 'images' / 'val',
        base_path / 'labels' / 'train',
        base_path / 'labels' / 'val'
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def partition_dataset(images_path, train_ratio=0.8):
    # List all image files
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.tif'))]
    random.shuffle(image_files)

    # Calculate split index
    train_size = int(len(image_files) * train_ratio)
    
    # Split image files into training and validation sets
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]
    
    return train_files, val_files

def copy_files(files, src_images_path, src_labels_path, dst_images_path, dst_labels_path):
    for file in files:
        # Copy image file
        shutil.copy(src_images_path / file, dst_images_path / file)
        
        # Copy corresponding label file
        label_file = Path(file).with_suffix('.txt')
        src_label_file = src_labels_path / label_file
        dst_label_file = dst_labels_path / label_file

        if src_label_file.exists():
            shutil.copy(src_label_file, dst_label_file)
        else:
            print(f"Warning: Label file {src_label_file} not found")

def main():
    base_path = Path('dataset')
    images_path = base_path / 'resized_plant'
    labels_path = base_path / 'annotations'
    
    # Create directory structure for train/val split
    create_folders(base_path)
    
    # Partition dataset into training and validation sets
    train_files, val_files = partition_dataset(images_path)
    
    # Copy training files
    copy_files(train_files, images_path, labels_path, base_path / 'images' / 'train', base_path / 'labels' / 'train')
    
    # Copy validation files
    copy_files(val_files, images_path, labels_path, base_path / 'images' / 'val', base_path / 'labels' / 'val')

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

if __name__ == '__main__':
    main()
