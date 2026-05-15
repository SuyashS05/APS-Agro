import os

# Path to label files
data_dir = "/content/drive/MyDrive/lane_detection_brc/YoLo/labels/Train"

# Class mapping: old -> new
class_mapping = {1: 0, 2: 1, 3: 2}

# Process each label file
for file in os.listdir(data_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(data_dir, file)
        new_lines = []
        
        with open(file_path, "r") as f:
            lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                old_class = int(parts[0])
                if old_class in class_mapping:
                    parts[0] = str(class_mapping[old_class])
                    new_lines.append(" ".join(parts))
                
        # Overwrite file with updated labels
        with open(file_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

print("Class labels updated successfully!")
