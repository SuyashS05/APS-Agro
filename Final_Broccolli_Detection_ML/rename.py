import os

def rename_images_in_folder(folder_path, prefix="val"):
    """
    Renames all images in the specified folder sequentially with a given prefix.

    Parameters:
        folder_path (str): Path to the folder containing the images.
        prefix (str): Prefix for the new image names.
    """
    try:
        # Get all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]

        # Sort files to maintain consistent ordering
        image_files.sort()

        # Rename each image file
        for i, file_name in enumerate(image_files, start=1):
            old_path = os.path.join(folder_path, file_name)
            new_name = f"{prefix}{i}.jpg"  # Using .jpg as default extension
            new_path = os.path.join(folder_path, new_name)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed {file_name} to {new_name}")
        
        print("Renaming completed successfully.")
    except Exception as e:
        print(f"Error renaming images: {e}")

# Example usage
if __name__ == "__main__":
    folder = r"F:\Final_Broccolli_Detection_ML\Dataset\Images\val"  # Replace with your folder path
    rename_images_in_folder(folder)
