import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

def convert_yolo_to_density(yolo_file, image_width, image_height):
    with open(yolo_file, 'r') as f:
        lines = f.readlines()
    
    points = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0]) 
        if class_id == 0:
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            points.append([x_center, y_center])
    
    return np.array(points)

def get_density_map_gaussian(im, points, sigma=15):
    h, w = im.shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)
    for point in points:
        x, y = min(w, max(1, int(point[0]))), min(h, max(1, int(point[1])))
        density_map[y-1, x-1] = 1
    density_map = gaussian_filter(density_map, sigma=sigma)
    return density_map

yolo_annotations_folder = '/home/edge/work/datasets/PANDA_dataset/labels/val'
image_folder = '/home/edge/work/datasets/PANDA_dataset/images/val'
output_csv_folder = '/home/edge/work/datasets/PANDA_dataset/csv_labels/val'
os.makedirs(output_csv_folder, exist_ok=True)

yolo_files = [f for f in os.listdir(yolo_annotations_folder) if f.endswith('.txt')]
for yolo_file in tqdm(yolo_files, desc="Processing files"):
    if yolo_file.endswith('.txt'):
        if yolo_file.split('.')[0] == "classes":
            continue
        output_csv_file = os.path.join(output_csv_folder, yolo_file.replace('.txt', '.csv'))
        if os.path.exists(output_csv_file):
            continue
        
        image_name = yolo_file.replace('.txt', '.jpg')
        image_path = os.path.join(image_folder, image_name)
        im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f'Image {image_path} not found.')
            continue
        
        h, w = im.shape[:2]
        points = convert_yolo_to_density(os.path.join(yolo_annotations_folder, yolo_file), w, h)
        
        if len(points) > 0:
            density_map = get_density_map_gaussian(im, points)
            
            np.savetxt(output_csv_file, density_map, delimiter=",")
