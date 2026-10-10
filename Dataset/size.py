from PIL import Image
import os

def get_image_resolutions(image_dir):
    resolutions = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):
            image_path = os.path.join(image_dir, filename)
            with Image.open(image_path) as img:
                width, height = img.size
                resolutions[filename] = (width, height)
    return resolutions

image_dir = 'dataset/images/test'
resolutions = get_image_resolutions(image_dir)
for filename, resolution in resolutions.items():
    print(f'{filename}: {resolution}')
