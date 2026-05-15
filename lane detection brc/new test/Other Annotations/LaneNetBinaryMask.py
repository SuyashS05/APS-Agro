import os
import numpy as np
import cv2

def create_lane_mask(annotation_folder, image_folder, output_folder):
    """Generate binary lane masks for LaneNet"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(annotation_folder):
        if filename.endswith(".txt"):
            image_path = os.path.join(image_folder, filename.replace(".txt", ".jpg"))  # Adjust for actual extension
            if not os.path.exists(image_path):
                print(f"Skipping {filename}: Image not found")
                continue

            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)

            annotation_path = os.path.join(annotation_folder, filename)
            with open(annotation_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    _, x1, y1, x2, y2 = map(int, parts)
                    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=2)

            mask_path = os.path.join(output_folder, filename.replace(".txt", ".png"))
            cv2.imwrite(mask_path, mask)
            print(f"Mask saved: {mask_path}")

# Example usage
create_lane_mask(
    annotation_folder="F:/final_Broccoli_Lane_detection_ML/Annotation/lane_Annotation/first_test_annotation/anotations/Test",
    image_folder="F:/final_Broccoli_Lane_detection_ML/Annotation/lane_Annotation/first_test_annotation/images/Test",
    output_folder="F:/final_Broccoli_Lane_detection_ML/Annotation/lane_Annotation/Other Annotations/lanenet/lanenet_masks/Test"
)
