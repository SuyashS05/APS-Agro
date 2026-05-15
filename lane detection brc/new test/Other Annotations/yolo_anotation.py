import os

def convert_to_yolo_format(input_folder, output_folder, image_width, image_height):
    """Convert annotation format to YOLO format"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, "r") as infile, open(output_path, "w") as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # Skip invalid lines
                    
                    class_id, x1, y1, x2, y2 = map(int, parts)
                    
                    x_center = (x1 + x2) / 2 / image_width
                    y_center = (y1 + y2) / 2 / image_height
                    width = abs(x2 - x1) / image_width
                    height = abs(y2 - y1) / image_height

                    outfile.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            print(f"Converted: {filename}")

# Example usage
convert_to_yolo_format(
    input_folder="F:/final_Broccoli_Lane_detection_ML/Annotation/lane_Annotation/first_test_annotation/anotations/Test",
    output_folder="F:/final_Broccoli_Lane_detection_ML/Annotation/lane_Annotation/Other Annotations/yolo/yolo_annotations/Test",
    image_width=800,  # Change based on actual image size
    image_height=800
)
