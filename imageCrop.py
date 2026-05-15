import os
from PIL import Image

def crop_to_square(input_folder, output_folder):
    """
    Crop images to 1:1 aspect ratio (center-cropped) and save them to the output folder.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where cropped images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if the file is an image
        if not (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')):
            continue

        try:
            with Image.open(input_path) as img:
                width, height = img.size
                # Calculate the crop box
                min_side = min(width, height)
                left = (width - min_side) // 2
                top = (height - min_side) // 2
                right = left + min_side
                bottom = top + min_side

                # Crop and save
                cropped_img = img.crop((left, top, right, bottom))
                output_path = os.path.join(output_folder, filename)
                cropped_img.save(output_path)
                print(f"Cropped and saved: {output_path}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

if __name__ == "__main__":
    source_folder = r"F:\brocolli_detection\dataset\images\test2Crop"
    destination_folder = r"F:\brocolli_detection\dataset\resized\test2Crop"

    crop_to_square(source_folder, destination_folder)
    print("Cropping completed!")
