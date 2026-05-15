import cv2
import os
import tkinter as tk
from tkinter import filedialog, simpledialog
import numpy as np

# Constants for line colors
MAIN_ROW_COLOR = (255, 255, 255)  # White
ADJACENT_ROW_COLOR = (255, 0, 0)  # Blue
APS_PATH_COLOR = (0, 0, 255)      # Red

# Global variables
annotations = []
current_image_index = 0
image_files = []
image_folder = ""
label_folder = ""
image = None
image_copy = None
start_x, start_y, end_x, end_y = -1, -1, -1, -1
drawing = False

def select_folder(prompt_message):
    """Prompt the user to select a folder."""
    folder_path = filedialog.askdirectory(title=prompt_message)
    return folder_path

def save_annotations(label_folder, image_name, annotations, image_size):
    """Save annotations to a text file in YOLO-like format."""
    txt_file = os.path.join(label_folder, f"{os.path.splitext(image_name)[0]}.txt")
    with open(txt_file, 'w') as file:
        for annotation in annotations:
            class_id, start_x, start_y, end_x, end_y = annotation
            file.write(f"{class_id} {start_x} {start_y} {end_x} {end_y}\n")
    print(f"Annotations saved to {txt_file}")

def draw_line(event, x, y, flags, param):
    """Handle mouse events to draw lines."""
    global start_x, start_y, end_x, end_y, drawing, image, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a line
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the line as the mouse moves
            image = image_copy.copy()
            cv2.line(image, (start_x, start_y), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish the line and prompt for class ID
        drawing = False
        end_x, end_y = x, y

        # Prompt for class ID
        class_id = simpledialog.askinteger("Class ID", "Enter class ID (1: Main Row, 2: Adjacent Row, 3: APS Path):")
        if class_id is not None:
            color = (
                MAIN_ROW_COLOR if class_id == 1 else
                ADJACENT_ROW_COLOR if class_id == 2 else
                APS_PATH_COLOR if class_id == 3 else
                (0, 255, 0)
            )
            # Save annotation and draw final line on the image
            annotations.append((class_id, start_x, start_y, end_x, end_y))
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)

def load_image():
    """Load the current image."""
    global image, image_copy, annotations
    image_path = os.path.join(image_folder, image_files[current_image_index])
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Resize image to fit the window
    image = cv2.resize(original_image, (800, 800))
    image_copy = image.copy()
    annotations.clear()
    print(f"Loaded: {image_files[current_image_index]}")

def handle_keypress():
    """Handle keypress events for saving and navigating."""
    global current_image_index, annotations

    while True:
        cv2.imshow("Image Annotation", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save annotations and move to the next image
            save_annotations(label_folder, image_files[current_image_index], annotations, image.shape[:2])
            current_image_index += 1
            if current_image_index >= len(image_files):
                print("No more images to annotate.")
                break
            load_image()

        elif key == ord('n'):  # Move to the next image without saving
            current_image_index += 1
            if current_image_index >= len(image_files):
                print("No more images to annotate.")
                break
            load_image()

        elif key == ord('p'):  # Move to the previous image
            current_image_index -= 1
            if current_image_index < 0:
                current_image_index = 0
            load_image()

        elif key == ord('q'):  # Quit the application
            print("Exiting annotation process.")
            break

def main():
    """Main function to run the annotation tool."""
    global image_folder, label_folder, image_files

    # Initialize Tkinter for folder selection
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Select folders
    image_folder = select_folder("Select the folder containing images:")
    label_folder = select_folder("Select the folder to save annotations:")

    if not image_folder or not label_folder:
        print("Both image and label folders are required.")
        return

    # Get the list of images
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No images found in the selected folder.")
        return

    # Load the first image and start the annotation process
    load_image()
    cv2.namedWindow("Image Annotation")
    cv2.setMouseCallback("Image Annotation", draw_line)
    handle_keypress()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
