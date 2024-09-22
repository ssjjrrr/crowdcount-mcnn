import os
import sys
import torch
import time
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader, NoGTDataLoader
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

data_path =  '/home/edge/work/datasets/PANDA_dataset/images/train_resized_960x540'
# gt_path = '/home/edge/work/datasets/PANDA_dataset/csv_labels/val'
model_path = './final_models/mcnn_shtechB_110.h5'
resolution = data_path.split('/')[-1].split('_')[-1]

output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()
mae = 0.0
mse = 0.0

#load test data
data_loader = NoGTDataLoader(data_path, shuffle=False, pre_load=False)
time_intervals = []
time_start = time.time()

for blob in data_loader:
    time_intervals.append(time.time() - time_start)
    print(time_intervals[-1])
    time_start = time.time()
    im_data = blob['data']
    # gt_data = blob['gt_density']
    # breakpoint()
    density_map = net(im_data)
    density_map = density_map.data.cpu().numpy()
    # gt_count = np.sum(gt_data)
    # et_count = np.sum(density_map)
    # mae += abs(gt_count-et_count)
    # mse += ((gt_count-et_count)*(gt_count-et_count))
    # if vis:
    #     utils.display_results(im_data, gt_data, density_map)
    if save_output:
        utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '_' + resolution + '.png')

time_intervals = np.array(time_intervals)
print(f'Time per image: {np.mean(time_intervals):0.4f}')

mae = mae/data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples())
print('\nMAE: %0.2f, MSE: %0.2f' % (mae,mse))

f = open(file_results, 'w') 
f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
f.close()