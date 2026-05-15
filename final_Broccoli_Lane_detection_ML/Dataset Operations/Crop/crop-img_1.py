import os
from PIL import Image

# Define paths
input_folder = r"F:\final_Broccoli_Lane_detection_ML\Dataset\Main-dataset\original\Train"
output_folder = r"F:\final_Broccoli_Lane_detection_ML\Dataset Operations\Crop\Cropped"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        
        width, height = img.size
        crop_size = width  # Square size (width x width)
        
        if height >= crop_size:
            # Top crop
            top_crop = img.crop((0, 0, width, crop_size))
            top_crop.save(os.path.join(output_folder, f"{filename}_top.jpg"))
            
            # Center crop
            center_y = (height - crop_size) // 2
            center_crop = img.crop((0, center_y, width, center_y + crop_size))
            center_crop.save(os.path.join(output_folder, f"{filename}_center.jpg"))
            
            # Bottom crop
            bottom_crop = img.crop((0, height - crop_size, width, height))
            bottom_crop.save(os.path.join(output_folder, f"{filename}_bottom.jpg"))
        else:
            print(f"Skipping {filename}: height ({height}) is smaller than width ({width}).")

print("Cropping complete.")
