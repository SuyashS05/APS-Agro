import os
import shutil

# Source and destination directories
source_dir = r"F:\lane detection brc\first_test_annotation\yolo_annotation\Train"
destination_dir = r"F:\lane detection brc\YoLo\labels\Train"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Get all image files in the source directory
image_files = sorted(os.listdir(source_dir))

# Rename and move images sequentially
for idx, filename in enumerate(image_files, start=1):
    # Get file extension (jpg, png, etc.)
    ext = os.path.splitext(filename)[1].lower()
    
    # Skip non-image files
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".txt"]:
        continue
    
    # New filename (e.g., 1.jpg, 2.png, ...)
    new_filename = f"{idx}{ext}"
    
    # Full paths
    old_path = os.path.join(source_dir, filename)
    new_path = os.path.join(destination_dir, new_filename)

    # Move and rename the file
    shutil.move(old_path, new_path)

print("Renaming and moving completed successfully!")
