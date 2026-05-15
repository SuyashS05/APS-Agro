import os
from PIL import Image

def compress_images_in_folder(input_folder, output_folder, quality=30, target_size_kb=300, resize_step=0.9):
    """
    Compresses all images in a folder and saves them in the output folder.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save compressed images.
        quality (int): Quality of the compressed images (1-100). Lower values mean higher compression.
        target_size_kb (int): Target size in kilobytes for the compressed images.
        resize_step (float): Factor by which to reduce image dimensions iteratively (e.g., 0.9 = reduce by 10%).
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

                # Open the image
                with Image.open(input_path) as img:
                    # Convert to RGB if necessary
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # Compress and resize iteratively until the target size is reached
                    current_quality = quality
                    while True:
                        # Save the image to a temporary file
                        temp_path = os.path.join(output_folder, "temp.jpg")
                        img.save(temp_path, "JPEG", optimize=True, quality=current_quality)

                        # Check file size
                        file_size_kb = os.path.getsize(temp_path) / 1024
                        if file_size_kb <= target_size_kb or current_quality <= 10:
                            # Rename the temp file to the output file
                            os.rename(temp_path, output_path)
                            print(f"Compressed {file_name} to {file_size_kb:.2f} KB and saved to {output_path}")
                            break
                        
                        # Reduce dimensions slightly if file size is too large
                        img = img.resize(
                            [int(dim * resize_step) for dim in img.size],
                            Image.Resampling.LANCZOS  # Updated from ANTIALIAS
                        )
                        
                        # Reduce quality further
                        current_quality = max(current_quality - 5, 10)
            else:
                print(f"Skipped non-image file: {file_name}")

    except Exception as e:
        print(f"Error processing folder: {e}")

# Example usage
if __name__ == "__main__":
    input_dir = "dataset/images/test"  # Replace with your input folder path
    output_dir = "dataset/resized/test2Crop"  # Replace with your output folder path
    target_kb = 300  # Target file size in kilobytes
    compress_images_in_folder(input_dir, output_dir, quality=30, target_size_kb=target_kb)
