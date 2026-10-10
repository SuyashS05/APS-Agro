import os
import cv2
import albumentations as A

# Paths for the dataset
image_folder = r"F:\Final_Broccolli_Detection_ML\Dataset\Images\val\val5"
label_folder = r"F:\Final_Broccolli_Detection_ML\Dataset\Labels\val\val5"
aug_image_folder = r"F:\Final_Broccolli_Detection_ML\ogmentation\images\val\train5"
aug_label_folder = r"F:\Final_Broccolli_Detection_ML\ogmentation\lables\val\train5"

# Create directories for augmented data if they don't exist
os.makedirs(aug_image_folder, exist_ok=True)
os.makedirs(aug_label_folder, exist_ok=True)

# Define augmentations
augmentations = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.HueSaturationValue(p=0.2),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
)

# Function to read YOLO annotation
def read_yolo_annotation(txt_file):
    bboxes, class_labels = [], []
    with open(txt_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split()
            class_labels.append(int(values[0]))
            bboxes.append([float(x) for x in values[1:]])
    return bboxes, class_labels

# Function to write YOLO annotation
def write_yolo_annotation(txt_file, bboxes, class_labels):
    with open(txt_file, "w") as file:
        for bbox, label in zip(bboxes, class_labels):
            file.write(f"{label} {' '.join(map(str, bbox))}\n")

# Clip bounding boxes to ensure they remain within valid bounds
def clip_bbox(bbox):
    x_center, y_center, width, height = bbox
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    return [x_center, y_center, width, height]

# Process images and annotations
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    label_path = os.path.join(label_folder, f"{os.path.splitext(image_file)[0]}.txt")

    # Load image and annotation
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    bboxes, class_labels = read_yolo_annotation(label_path)

    # Generate augmented data
    for aug_idx in range(5):  # Generate 5 augmented versions per image
        try:
            augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented["image"]
            aug_bboxes = [clip_bbox(bbox) for bbox in augmented["bboxes"]]
            aug_class_labels = augmented["class_labels"]

            # Save augmented image
            aug_image_name = f"aug_{idx}_{aug_idx}.jpg"
            aug_label_name = f"aug_{idx}_{aug_idx}.txt"
            aug_image_path = os.path.join(aug_image_folder, aug_image_name)
            aug_label_path = os.path.join(aug_label_folder, aug_label_name)

            cv2.imwrite(aug_image_path, aug_image)
            write_yolo_annotation(aug_label_path, aug_bboxes, aug_class_labels)

        except Exception as e:
            print(f"Error augmenting {image_file}: {e}")
            continue

    print(f"Processed {idx + 1}/{len(image_files)}: {image_file}")

print("Augmentation completed!")
