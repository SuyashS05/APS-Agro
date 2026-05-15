from PIL import Image
import os

# Define the folder path
folder_path = r"E:\DreamProject\FreshAGRO_Fire\freshagro_userfire\assets\productType"

# Loop through all files in the directory
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Check if it's an image (excluding .png to avoid reconversion)
    if filename.lower().endswith(('.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
        # Open the image
        with Image.open(file_path) as img:
            # Define the new filename with .png extension
            new_filename = os.path.splitext(filename)[0] + ".png"
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Convert and save as PNG
            img.save(new_file_path, "PNG")
            print(f"Converted: {filename} -> {new_filename}")
        
        # Remove the original file
        os.remove(file_path)
        print(f"Deleted: {filename}")
    
    # Remove any non-PNG files
    elif not filename.lower().endswith('.png'):
        os.remove(file_path)
        print(f"Deleted non-image file: {filename}")
