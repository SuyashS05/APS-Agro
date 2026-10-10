import cv2
import os
import tkinter as tk
from tkinter import filedialog, simpledialog

# Global variables
annotations = []
current_image_index = 0
start_x, start_y, end_x, end_y = -1, -1, -1, -1
drawing = False
image_files = []
image = None
image_copy = None
image_width, image_height = 0, 0
window_width, window_height = 800, 600  # Set the window size (adjust as needed)

def select_folder(prompt_message):
    """Prompt the user to select a folder."""
    folder_path = filedialog.askdirectory(title=prompt_message)
    return folder_path

def save_annotations(label_folder, image_name, annotations):
    """Save annotations to a text file in YOLO format."""
    txt_file = os.path.join(label_folder, f"{os.path.splitext(image_name)[0]}.txt")
    with open(txt_file, 'w') as file:
        for annotation in annotations:
            file.write(" ".join(map(str, annotation)) + "\n")
    print(f"Annotations saved to {txt_file}")

def draw_rectangle(event, x, y, flags, param):
    """Handle mouse events to draw bounding boxes."""
    global start_x, start_y, end_x, end_y, drawing, annotations, image, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x, end_y = x, y
            image = image_copy.copy()
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y

        # Get class number
        class_num = simpledialog.askinteger("Class Number", "Enter class number:")
        if class_num is not None:
            # Convert to YOLO format (relative to original image size)
            x_center = (start_x + end_x) / 2 / window_width  # Use window size for display
            y_center = (start_y + end_y) / 2 / window_height  # Use window size for display
            width = abs(end_x - start_x) / window_width  # Use window size for display
            height = abs(end_y - start_y) / window_height  # Use window size for display
            annotations.append([class_num, x_center, y_center, width, height])
            print(f"Annotation added: {annotations[-1]}")

        # Draw final rectangle on the image
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

def load_image():
    """Load the current image."""
    global image, image_copy, image_width, image_height, annotations
    image_path = os.path.join(image_folder, image_files[current_image_index])
    original_image = cv2.imread(image_path)
    image_height, image_width = original_image.shape[:2]

    # Resize image to fit the window
    image = cv2.resize(original_image, (window_width, window_height))
    image_copy = image.copy()
    annotations.clear()
    print(f"Loaded: {image_files[current_image_index]}")

def show_image():
    """Show the current image with annotations."""
    global image
    temp_image = image_copy.copy()
    for annotation in annotations:
        class_num, x_center, y_center, width, height = annotation
        # Convert back to original image dimensions
        start_x = int((x_center - width / 2) * window_width)  # Convert to window size
        start_y = int((y_center - height / 2) * window_height)  # Convert to window size
        end_x = int((x_center + width / 2) * window_width)  # Convert to window size
        end_y = int((y_center + height / 2) * window_height)  # Convert to window size
        # Draw the bounding box in the window
        cv2.rectangle(temp_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(temp_image, str(class_num), (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Image Annotation", temp_image)

def handle_keypress(label_folder):
    """Handle keypress events for navigation and saving."""
    global current_image_index, annotations
    while True:
        show_image()
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save and move to next image
            save_annotations(label_folder, image_files[current_image_index], annotations)
            current_image_index += 1
            if current_image_index >= len(image_files):
                print("No more images to annotate.")
                break
            load_image()
        elif key == ord('n'):  # Skip to the next image
            current_image_index += 1
            if current_image_index >= len(image_files):
                print("No more images to annotate.")
                break
            load_image()
        elif key == ord('p'):  # Go back to the previous image
            current_image_index -= 1
            if current_image_index < 0:
                current_image_index = 0
            load_image()
        elif key == ord('q'):  # Quit
            print("Annotation process terminated.")
            break

# Main application logic
root = tk.Tk()
root.withdraw()  # Hide the root window

# Prompt user for folder paths
image_folder = select_folder("Select the folder containing images:")
label_folder = select_folder("Select the folder to save YOLO annotations:")

if image_folder and label_folder:
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("No images found in the selected folder.")
    else:
        load_image()
        cv2.namedWindow("Image Annotation")
        cv2.setMouseCallback("Image Annotation", draw_rectangle)
        handle_keypress(label_folder)

    cv2.destroyAllWindows()
else:
    print("Both image and label folder paths are required.")
