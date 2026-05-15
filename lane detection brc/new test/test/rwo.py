import cv2
import json
import os
import numpy as np

# Directory containing images to annotate
image_dir = "dataset/row detect/img"
# Directory to save annotations
output_dir = "pdataset/row detect/annotation"
os.makedirs(output_dir, exist_ok=True)

# Square size for the annotation screen
annotation_size = 800

# Annotation function
def annotate_image(image_path, output_path):
    annotations = {"main_row": [], "adjacent_rows": [], "aps_path": []}

    # Load the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Resize image to square with padding
    max_dim = max(h, w)
    padded_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    padded_image[:h, :w, :] = image
    resized_image = cv2.resize(padded_image, (annotation_size, annotation_size))
    clone = resized_image.copy()

    cv2.namedWindow("Annotate Image")

    # State variables
    current_line = None
    annotation_type = "main_row"  # Default to main row

    # Mouse callback function
    def draw_line(event, x, y, flags, param):
        nonlocal current_line, annotations, annotation_type
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing line
            current_line = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing line
            current_line.append((x, y))
            x1, y1 = current_line[0]
            x2, y2 = current_line[1]

            # Define line color based on type
            if annotation_type == "main_row":
                color = (255, 255, 255)  # White
                annotations["main_row"].append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                annotation_type = "adjacent_rows"  # Switch to adjacent rows
            elif annotation_type == "adjacent_rows":
                color = (255, 0, 0)  # Blue
                annotations["adjacent_rows"].append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            elif annotation_type == "aps_path":
                color = (0, 0, 255)  # Red
                annotations["aps_path"].append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

            # Draw the line
            cv2.line(clone, (x1, y1), (x2, y2), color, 2)

            # Reset current line
            current_line = None

    # Set the mouse callback
    cv2.setMouseCallback("Annotate Image", draw_line)

    while True:
        cv2.imshow("Annotate Image", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):  # Reset annotations
            clone = resized_image.copy()
            annotations = {"main_row": [], "adjacent_rows": [], "aps_path": []}
            annotation_type = "main_row"
        elif key == ord("n"):  # Switch to next annotation type
            if annotation_type == "main_row":
                annotation_type = "adjacent_rows"
            elif annotation_type == "adjacent_rows":
                annotation_type = "aps_path"
            elif annotation_type == "aps_path":
                annotation_type = "main_row"
        elif key == ord("s"):  # Save annotations and move to the next image
            break
        elif key == ord("q"):  # Quit annotation
            cv2.destroyAllWindows()
            exit()

    # Save annotations as JSON
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=4)

    cv2.destroyAllWindows()

# Iterate through images in the directory
for image_file in os.listdir(image_dir):
    if image_file.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_dir, image_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}.json")
        print(f"Annotating {image_file}...")
        annotate_image(image_path, output_path)

print("Annotation complete!")
