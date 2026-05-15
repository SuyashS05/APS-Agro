import cv2
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
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
    if not annotations:
        print(f"Skipping {image_name}: No annotations.")
        return
    
    os.makedirs(label_folder, exist_ok=True)  # Ensure output folder exists
    txt_file = os.path.join(label_folder, f"{os.path.splitext(image_name)[0]}.txt")
    
    try:
        with open(txt_file, 'w') as file:
            for annotation in annotations:
                class_id, start_x, start_y, end_x, end_y = annotation
                file.write(f"{class_id} {start_x} {start_y} {end_x} {end_y}\n")
        print(f"✅ Annotations saved to {txt_file}")
    except Exception as e:
        print(f"❌ Error saving annotations: {e}")

def draw_line(event, x, y, flags, param):
    """Handle mouse events to draw lines."""
    global start_x, start_y, end_x, end_y, drawing, image, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            print("❌ Click outside the image bounds.")
            return  # Prevent drawing outside image
        
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        image = image_copy.copy()
        cv2.line(image, (start_x, start_y), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        if not drawing:
            return

        drawing = False
        end_x, end_y = x, y

        # Prompt for class ID with better error handling
        while True:
            class_id = simpledialog.askinteger("Class ID", "Enter class ID (1: Main Row, 2: Adjacent Row, 3: APS Path):")
            if class_id in [1, 2, 3]:
                break
            elif class_id is None:  # User canceled
                return
            else:
                messagebox.showerror("Invalid Input", "Please enter 1, 2, or 3.")

        # Choose color based on class
        color = {1: MAIN_ROW_COLOR, 2: ADJACENT_ROW_COLOR, 3: APS_PATH_COLOR}.get(class_id, (0, 255, 0))

        # Save annotation and draw final line on the image
        annotations.append((class_id, start_x, start_y, end_x, end_y))
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)

def load_image():
    """Load the current image."""
    global image, image_copy, annotations
    image_path = os.path.join(image_folder, image_files[current_image_index])
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ Cannot load image: {image_path}")
        return False

    # Resize image to fit the window
    image = cv2.resize(original_image, (800, 800))
    image_copy = image.copy()
    annotations.clear()
    print(f"📷 Loaded: {image_files[current_image_index]}")
    return True

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
                print("✅ All images annotated. Exiting.")
                break
            if not load_image():
                break

        elif key == ord('n'):  # Skip saving, move to next
            current_image_index += 1
            if current_image_index >= len(image_files):
                print("✅ No more images.")
                break
            if not load_image():
                break

        elif key == ord('p'):  # Previous image
            if current_image_index > 0:
                current_image_index -= 1
                load_image()
            else:
                print("🚫 No previous image.")

        elif key == ord('q'):  # Quit
            print("🔴 Exiting annotation process.")
            break

def main():
    """Main function to run the annotation tool."""
    global image_folder, label_folder, image_files

    # Initialize Tkinter for folder selection
    root = tk.Tk()
    root.withdraw()

    image_folder = select_folder("Select the folder containing images:")
    label_folder = select_folder("Select the folder to save annotations:")

    if not image_folder or not label_folder:
        print("❌ Both image and label folders are required.")
        return

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("❌ No images found in the selected folder.")
        return

    if not load_image():
        return

    cv2.namedWindow("Image Annotation")
    cv2.setMouseCallback("Image Annotation", draw_line)
    handle_keypress()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
