import os
import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Step 1: Load the Trained Model
model_path = 'best.pt'  # Path to your trained model weights
model = YOLO(model_path)

# Step 2: Use a File Dialog to Select an Image or Video
def select_file(file_type="image"):
    # Create a tkinter window (we don't want to show the root window)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select an image or video file based on file_type
    file_types = [("Image Files", ".jpg;.jpeg;.png;.bmp")] if file_type == "image" else [("Video Files", ".mp4;.avi;.mov;.mkv")]
    file_path = filedialog.askopenfilename(
        title=f"Select a {file_type.capitalize()}",
        filetypes=file_types,  # Supported formats
        initialdir=os.getcwd()  # Start from the current working directory
    )
    
    return file_path

# Step 3: Test on the Selected Image or Video
def test_media(media_path, media_type="image"):
    if media_type == "image":
        # Read the image
        img = cv2.imread(media_path)
        
        # Inference on the image
        results = model(img)

        # Extract the first result (this is a Detection object)
        result = results[0]

        # Display results
        result.show()  # This will show the image with detected boxes
        
        # Save the image with detections
        result.save()  # This will save the image with detections in the current directory

        # You can also plot the image if you want to display in a notebook
        plt.imshow(cv2.cvtColor(result.ims[0], cv2.COLOR_BGR2RGB))
        plt.show()

    elif media_type == "video":
        # Open the video file
        cap = cv2.VideoCapture(media_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference on the current frame
            results = model(frame)

            # Extract the first result from the list (this will be a Detection object)
            result = results[0]

            # Display results on frame
            frame_with_results = result.plot()  # Get the frame with detections
            
            # Show the frame
            cv2.imshow("YOLOv8 Detection", frame_with_results)
            
            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Step 4: Test on Webcam
def test_camera():
    cap = cv2.VideoCapture(0)  # Use the default camera (usually the first one)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inference on the current frame
        results = model(frame)

        # Extract the first result from the list (this will be a Detection object)
        result = results[0]

        # Display results on frame
        frame_with_results = result.plot()  # Get the frame with detections
        
        # Show the frame
        cv2.imshow("YOLOv8 Detection", frame_with_results)
        
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Step 5: Main Execution
if __name__ == "__main__":
    # Ask the user to select the mode
    mode = input("Choose mode: 1 for Image, 2 for Webcam, 3 for Video: ")

    if mode == "1":
        # Allow the user to select an image file
        image_path = select_file(file_type="image")
        
        if image_path:
            print(f"Selected Image: {image_path}")
            # Test the model on the selected image
            test_media(image_path, media_type="image")
        else:
            print("No image selected.")
    
    elif mode == "2":
        # Test the model using webcam
        print("Starting Webcam for detection...")
        test_camera()

    elif mode == "3":
        # Allow the user to select a video file
        video_path = select_file(file_type="video")
        
        if video_path:
            print(f"Selected Video: {video_path}")
            # Test the model on the selected video
            test_media(video_path, media_type="video")
        else:
            print("No video selected.")

    else:
        print("Invalid mode selected.")
