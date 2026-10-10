import os
from PIL import Image

def compress_images_in_folder(input_folder, output_folder, quality=30):
    """
    Compresses all images in a folder and saves them in the output folder.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save compressed images.
        quality (int): Quality of the compressed images (1-100). Lower values mean higher compression.
    """
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Loop through all files in the input folder
        for file_name in os.listdir(input_folder):
            input_path = os.path.join(input_folder, file_name)

            # Check if the file is an image (by extension)
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                output_path = os.path.join(output_folder, file_name)

                # Open and compress the image
                with Image.open(input_path) as img:
                    # Convert to RGB if necessary
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # Save compressed image
                    img.save(output_path, "JPEG", optimize=True, quality=quality)
                    print(f"Compressed {file_name} and saved to {output_path}")
            else:
                print(f"Skipped non-image file: {file_name}")

    except Exception as e:
        print(f"Error processing folder: {e}")

# Example usage
if __name__ == "__main__":
    input_dir = "dataset/images/test"  # Replace with your input folder path
    output_dir = "dataset/images/test2Crop"  # Replace with your output folder path
    compression_quality = 20  # Set desired compression quality (1-100)

    compress_images_in_folder(input_dir, output_dir, compression_quality)