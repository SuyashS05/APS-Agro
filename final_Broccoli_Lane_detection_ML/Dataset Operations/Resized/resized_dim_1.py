import os
from PIL import Image

# Define paths
input_folder = r"F:\final_Broccoli_Lane_detection_ML\Dataset Operations\Crop\Cropped"
output_folder = r"F:\final_Broccoli_Lane_detection_ML\Dataset Operations\Resized\Resized"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process images
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

for idx, filename in enumerate(image_files, start=1):
    img_path = os.path.join(input_folder, filename)
    img = Image.open(img_path)
    
    # Resize to 640x640
    resized_img = img.resize((640, 640), Image.LANCZOS)
    
    # Save with sequential numbering
    save_path = os.path.join(output_folder, f"{idx}.jpg")
    resized_img.save(save_path, "JPEG")

print("Resizing complete.")
