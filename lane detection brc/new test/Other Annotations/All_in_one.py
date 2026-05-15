import os
import cv2
import numpy as np

# Input Directories
IMAGE_DIR = "F:/final_Broccoli_Lane_detection_ML/Annotation/lane_Annotation/first_test_annotation/images/Test"
ANNOTATIONS_DIR = "F:/final_Broccoli_Lane_detection_ML/Annotation/lane_Annotation/first_test_annotation/anotations/Test"

# Output Directories
OUTPUT_DIR = "F:/final_Broccoli_Lane_detection_ML/Annotation/lane_Annotation/Other Annotations/output_Test/"
os.makedirs(OUTPUT_DIR + "yolo/", exist_ok=True)
os.makedirs(OUTPUT_DIR + "lanenet_masks/", exist_ok=True)
os.makedirs(OUTPUT_DIR + "deeplab_masks/", exist_ok=True)

def convert_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert bbox (x1, y1, x2, y2) to YOLO format."""
    x_center = (x1 + x2) / (2 * img_w)
    y_center = (y1 + y2) / (2 * img_h)
    width = abs(x2 - x1) / img_w
    height = abs(y2 - y1) / img_h
    return x_center, y_center, width, height

def process_annotations():
    for txt_file in os.listdir(ANNOTATIONS_DIR):
        if not txt_file.endswith(".txt"):
            continue

        image_name = txt_file.replace(".txt", ".jpg")
        image_path = os.path.join(IMAGE_DIR, image_name)
        img = cv2.imread(image_path)

        if img is None:
            print(f"⚠ Warning: Image {image_name} not found. Skipping...")
            continue
        
        img_h, img_w, _ = img.shape
        mask_lanenet = np.zeros((img_h, img_w), dtype=np.uint8)  # LaneNet grayscale mask
        mask_deeplab = np.zeros((img_h, img_w, 3), dtype=np.uint8)  # DeepLabV3 RGB mask

        with open(os.path.join(ANNOTATIONS_DIR, txt_file), "r") as f:
            yolo_annotations = []
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"⚠ Skipping invalid line in {txt_file}: {line.strip()}")
                    continue

                class_id, x1, y1, x2, y2 = map(int, parts)

                # Convert for YOLO
                x_center, y_center, width, height = convert_to_yolo(x1, y1, x2, y2, img_w, img_h)
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # LaneNet Mask (White lanes on black background)
                cv2.line(mask_lanenet, (x1, y1), (x2, y2), color=255, thickness=3)

                # DeepLabV3 Mask (Unique color for each lane class)
                color = (class_id * 40 % 255, class_id * 80 % 255, class_id * 120 % 255)
                cv2.line(mask_deeplab, (x1, y1), (x2, y2), color=color, thickness=3)

        # Save YOLO annotations
        with open(os.path.join(OUTPUT_DIR, "yolo", txt_file), "w") as f:
            f.write("\n".join(yolo_annotations))

        # Save LaneNet mask (grayscale PNG)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "lanenet_masks", image_name.replace(".jpg", ".png")), mask_lanenet)

        # Save DeepLabV3 mask (RGB PNG)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "deeplab_masks", image_name.replace(".jpg", ".png")), mask_deeplab)

        print(f"✅ Processed: {image_name}")

if __name__ == "__main__":
    process_annotations()
    print("🎉 Conversion completed!")
