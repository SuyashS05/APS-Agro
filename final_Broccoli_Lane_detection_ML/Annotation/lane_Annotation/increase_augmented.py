import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import shutil

# Define paths
image_folder = r"F:\final_Broccoli_Lane_detection_ML\Annotation\lane_Annotation\first_test_annotation\images"
annotation_folder = r"F:\final_Broccoli_Lane_detection_ML\Annotation\lane_Annotation\first_test_annotation\anotations"
output_image_folder = r"F:\final_Broccoli_Lane_detection_ML\Annotation\lane_Annotation\augmented_dataset\images"
output_annotation_folder = r"F:\final_Broccoli_Lane_detection_ML\Annotation\lane_Annotation\augmented_dataset\annotations"

# Create output folders if they don’t exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_annotation_folder, exist_ok=True)

def augment_image(img):
    # Flip horizontally
    flipped = cv2.flip(img, 1)
    
    # Rotate 15 degrees
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), 15, 1.0)
    rotated = cv2.warpAffine(img, matrix, (width, height))
    
    # Brightness adjustment
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_img)
    brightened = np.array(enhancer.enhance(1.5))
    brightened = cv2.cvtColor(brightened, cv2.COLOR_RGB2BGR)
    
    return [flipped, rotated, brightened]

# Process images and annotations
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
        img_path = os.path.join(image_folder, filename)
        annotation_path = os.path.join(annotation_folder, filename.replace('.jpg', '.txt'))  # Assuming .txt annotation
        
        img = cv2.imread(img_path)
        augmented_images = augment_image(img)
        
        # Save augmented images and copy annotations
        base_name = os.path.splitext(filename)[0]
        for i, aug_img in enumerate(augmented_images, start=1):
            new_img_name = f"{base_name}_aug{i}.jpg"
            new_annotation_name = f"{base_name}_aug{i}.txt"
            cv2.imwrite(os.path.join(output_image_folder, new_img_name), aug_img)
            
            if os.path.exists(annotation_path):
                shutil.copy(annotation_path, os.path.join(output_annotation_folder, new_annotation_name))

print("Augmentation complete.")
