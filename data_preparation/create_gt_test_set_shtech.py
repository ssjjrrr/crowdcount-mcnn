import os
import cv2
import numpy as np
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter

def get_density_map_gaussian(im, points):
    h, w = im.shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)
    for point in points:
        x, y = min(w, max(1, int(point[0]))), min(h, max(1, int(point[1])))
        density_map[y-1, x-1] = 1
    density_map = gaussian_filter(density_map, sigma=15)
    return density_map

dataset = 'B'
dataset_name = f'shanghaitech_part_{dataset}'
path = f'../data/original/shanghaitech/part_{dataset}_final/test_data/images/'
gt_path = f'../data/original/shanghaitech/part_{dataset}_final/test_data/ground_truth/'
gt_path_csv = f'../data/original/shanghaitech/part_{dataset}_final/test_data/ground_truth_csv/'

os.makedirs(gt_path_csv, exist_ok=True)
num_images = 182 if dataset == 'A' else 316

for i in range(1, num_images + 1):
    if i % 10 == 0:
        print(f'Processing {i}/{num_images} files')
    
    mat_data = loadmat(os.path.join(gt_path, f'GT_IMG_{i}.mat'))
    input_img_name = os.path.join(path, f'IMG_{i}.jpg')
    im = cv2.imread(input_img_name, cv2.IMREAD_GRAYSCALE)
    if im is None:
        print(f'Image {input_img_name} not found.')
        continue

    annPoints = mat_data['image_info'][0,0]['location'][0,0]
    im_density = get_density_map_gaussian(im, annPoints)
    np.savetxt(os.path.join(gt_path_csv, f'IMG_{i}.csv'), im_density, delimiter=",")
