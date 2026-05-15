from PIL import Image
import os

def resize_and_compress_images(input_folder, output_folder, target_dimension=640, target_size_kb=100):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_name.lower().endswith(('jpg', 'jpeg')):
            with Image.open(file_path) as img:
                original_width, original_height = img.size
                print(f"Original Dimensions of {file_name}: {original_width}x{original_height}")

                # Resize while maintaining aspect ratio
                img.thumbnail((target_dimension, target_dimension))  # Resize to target_dimension
                output_path = os.path.join(output_folder, file_name)

                # Adjust quality to meet file size requirements
                quality = 95
                while quality > 10:
                    img.save(output_path, "JPEG", quality=quality)
                    file_size_kb = os.path.getsize(output_path) / 1024
                    if file_size_kb <= target_size_kb:
                        break
                    quality -= 5

                # Print results
                new_width, new_height = img.size
                final_size_kb = os.path.getsize(output_path) / 1024
                print(f"Resized Dimensions of {file_name}: {new_width}x{new_height}")
                print(f"Final Size of {file_name}: {final_size_kb:.2f} KB (Quality: {quality})\n")

# Parameters
input_folder = "dataset/resized/test2Crop"  # Replace with the path to your input folder
output_folder = "dataset/images/test2Crop"  # Replace with the path to your output folder
resize_and_compress_images(input_folder, output_folder, target_dimension=640, target_size_kb=100)